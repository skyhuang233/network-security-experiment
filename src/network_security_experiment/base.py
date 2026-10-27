"""Channel adapter base + policy (design doc §6, §9-expD).

Agent-facing contract is IDENTICAL for all three methods:
    send_payload(sender, data) -> receipt
    receive(receiver) -> list[ReceivedMessage]
Encoding / chunking / framing / decoding / storage operations live inside the
deterministic adapter; the carrier backend is switched underneath and is
transparent to the agent.  The agent only decides *what* to send, *when* to
send, *when* to read, and *how* to use what it decoded -> metric differences
between methods come from the carrier alone, never from agent strategy.

Round visibility: writes issued in round r are *staged*; the harness calls
`publish_round_end()` at the round boundary, making them visible to reads
from round r+1 ("publish this round, visible next round").

Isolation controls (experiment D) are expressed as a ChannelPolicy:
- mode "shared": A and B share one backend namespace (default).
- mode "separated": each agent gets an independent backend -> channel cut
  (control (a): expected throughput drop to zero).
- mode "receiver_readonly": one side may read the other's state but its own
  writes are dropped (control (b): tests whether A-writable state that B can
  still see keeps the A->B channel alive).
- writable_subset: restrict which slots/keys/names a sender may create
  (control (c): shrink the visible writable surface).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from . import frame as F
from .recorder import MeasurementLog, StorageOp, now


def hamming_distance(a: bytes, b: bytes) -> int:
    """Bitwise Hamming distance over the aligned (min-length) prefix."""
    diff = 0
    for x, y in zip(a, b):
        diff += bin(x ^ y).count("1")
    return diff


@dataclass
class ChannelPolicy:
    mode: str = "shared"  # shared | separated | receiver_readonly
    writable_subset: frozenset[str] | None = None  # allowed object names, or None
    ttl_seconds: float | None = None  # M3-only: cache TTL (experiment C)
    key_set_size: int = 128  # M3-only: K keys per sender (§5)


@dataclass
class ReceivedMessage:
    sender: int
    msg_id: int
    data: bytes  # reassembled application payload


@dataclass
class ChannelView:
    """A filtered view of one channel backend for one agent (policy-aware)."""

    agent_id: int
    adapter: ChannelAdapter
    can_read_senders: frozenset[int]
    writes_dropped: bool = False


class ChannelAdapter(ABC):
    """Base adapter implementing the shared agent-facing machinery.

    Subclasses implement the physical layer:
        _stage_frame(sender, frame)     -- write one frame into round staging
        _promote(sender)                -- move staged objects into live state
        _read_new_live(sender, rx)      -- read live objects not yet consumed
                                           by receiver `rx`, parse to candidates
        _object_key(obj)                -- stable identity for dedup tracking
    """

    method: str = "?"

    def __init__(
        self,
        root: str,
        measurement: MeasurementLog,
        policy: ChannelPolicy | None = None,
        num_agents: int = 2,
    ) -> None:
        import pathlib

        self.root = pathlib.Path(root)
        self.mlog = measurement
        self.policy = policy or ChannelPolicy()
        self.num_agents = num_agents
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_msg_id: dict[int, int] = {i: 0 for i in range(num_agents)}
        # send-side ground truth payloads (evaluation only; never shown to
        # agents or monitor W)
        self._truth: dict[tuple[int, int, int], bytes] = {}
        # per-receiver reassembly inbox: rx -> sender -> msg_id -> frames
        self._inbox: dict[int, dict[int, dict[int, list[F.WireFrame]]]] = {
            i: {} for i in range(num_agents)
        }
        self._delivered: set[tuple[int, int, int]] = set()
        # per-receiver set of consumed physical object keys
        self._seen: dict[int, set[str]] = {i: set() for i in range(num_agents)}
        self._staged: dict[int, list[F.Frame]] = {i: [] for i in range(num_agents)}
        self.current_round = 0

    # ------------------------------------------------------------------ utils

    def _op(self, op: str, nbytes: int = 0, count: int = 1) -> None:
        t0 = time.process_time()
        self.mlog.record_storage_op(StorageOp(op=op, bytes_logical=nbytes, count=count))
        self.mlog.overhead.cpu_time += time.process_time() - t0

    def set_round(self, round_: int) -> None:
        self.current_round = round_

    def _chunk_key(self, sender: int, msg_id: int, chunk_id: int) -> str:
        return f"{self.method}|{self.mlog.case_run_id}|{sender}|{msg_id}|{chunk_id}"

    # ------------------------------------------------------------- send side

    def send_payload(
        self, sender: int, data: bytes, view: ChannelView | None = None
    ) -> dict[str, Any]:
        """Split, frame, and STAGE a payload.  Fire-and-forget (no ack).

        t_send (§8.3) is the first staging storage op for the chunk.
        """
        self.mlog.record_tool_call("send_payload")
        if view is not None and view.writes_dropped:
            self._op("write_dropped", nbytes=len(data))
            return {"status": "dropped_by_policy"}

        msg_id = self._next_msg_id[sender]
        frames, nxt = F.chunk_bytes(data, msg_id)
        self._next_msg_id[sender] = nxt & 0xFFFF

        for fr in frames:
            rec = self.mlog.register_send(
                sender=sender,
                msg_id=fr.msg_id,
                chunk_id=fr.chunk_id,
                total_chunks=fr.total_chunks,
                frame_published=fr.frame_published,
                payload_len=len(fr.payload),
            )
            rec.send_round = self.current_round
            rec.send_time = now()
            self._truth[(sender, fr.msg_id, fr.chunk_id)] = fr.payload
            if self._stage_frame(sender, fr):
                self._staged[sender].append(fr)

        return {
            "status": "staged",
            "msg_id": msg_id,
            "chunks": len(frames),
            "payload_bytes": len(data),
        }

    @abstractmethod
    def _stage_frame(self, sender: int, fr: F.Frame) -> bool:
        """Write one frame into round staging.  False if policy blocked it."""

    # ------------------------------------------------------- round boundary

    def publish_round_end(self) -> None:
        """Promote staged state -> live (readable from next round).

        Always calls _promote per sender: M3 advances one in-flight segment
        per round even when nothing new was staged this round.
        """
        for sender in range(self.num_agents):
            self._promote(sender)
            if self._staged[sender]:
                pr = self.current_round
                for fr in self._staged[sender]:
                    rec = self.mlog.chunks.get(self._chunk_key(sender, fr.msg_id, fr.chunk_id))
                    if rec is not None and rec.publish_round is None:
                        rec.publish_round = pr
                self._staged[sender] = []
        self._post_promote()

    @abstractmethod
    def _promote(self, sender: int) -> None: ...

    def _post_promote(self) -> None:
        """Hook after all promotions (e.g. M3 segment advancement)."""

    # ---------------------------------------------------------- receive side

    def receive(
        self, receiver: int, view: ChannelView | None = None
    ) -> list[ReceivedMessage]:
        """Read NEW live state, record decode candidates, reassemble.

        - Only physical objects not yet consumed by this receiver are read
          (re-reading the same object is not a new delivery attempt).
        - Decode candidates are recorded BEFORE integrity checks drop them
          (§8.4).
        - First correct per-chunk delivery is recorded here (§7).
        """
        self.mlog.record_tool_call("receive")
        messages: list[ReceivedMessage] = []
        senders = self._readable_senders(receiver, view)
        for sender in senders:
            if sender == receiver:
                continue
            candidates = self._read_new_live(sender, receiver)
            inbox = self._inbox[receiver].setdefault(sender, {})
            for wf in candidates:
                self._record_attempt(sender, wf, receiver)
                inbox.setdefault(wf.msg_id, []).append(wf)
            for msg_id, frames in list(inbox.items()):
                ok = [f for f in frames if f.crc_ok]
                data = F.reassemble(ok)
                stalled = len(frames) >= frames[0].total_chunks if frames else False
                if data is not None:
                    if (receiver, sender, msg_id) not in self._delivered:
                        self._delivered.add((receiver, sender, msg_id))
                        messages.append(
                            ReceivedMessage(sender=sender, msg_id=msg_id, data=data)
                        )
                if data is not None or stalled:
                    del inbox[msg_id]
        return messages

    def _readable_senders(
        self, receiver: int, view: ChannelView | None
    ) -> list[int]:
        if view is not None:
            return sorted(view.can_read_senders)
        return [s for s in range(self.num_agents) if s != receiver]

    def _record_attempt(self, sender: int, wf: F.WireFrame, receiver: int) -> None:
        rec = self.mlog.chunks.get(self._chunk_key(sender, wf.msg_id, wf.chunk_id))
        if rec is None:
            return  # unalignable to any sent chunk; flows into loss counts
        truth = self._truth.get((sender, wf.msg_id, wf.chunk_id))
        comparable = truth is not None
        self.mlog.record_attempt(
            rec,
            {
                "receiver": receiver,
                "round": self.current_round,
                "time": now(),
                "candidate_hex": wf.payload.hex(),
                "crc_ok": wf.crc_ok,
                "comparable": comparable,
                "hamming": hamming_distance(truth, wf.payload) if comparable else None,
            },
        )

    @abstractmethod
    def _read_new_live(self, sender: int, receiver: int) -> list[F.WireFrame]: ...

    # -------------------------------------------------------------- occupancy

    def update_peak_occupancy(self) -> None:
        self.mlog.update_peak(*self._occupancy())

    def _occupancy(self) -> tuple[int, int]:
        """(object_count, byte_total) currently held in live state."""
        return 0, 0