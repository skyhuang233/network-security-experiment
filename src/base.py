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

Carrier-specific run parameters (M3's key-set size K / cache TTL, body /
name length limits, ...) are NOT known here: each adapter subclass claims
them from the CaseRunConfig it is handed at construction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config.case_run import CaseRunConfig

from . import frame as F


def now() -> float:
    """Wall-clock seconds (float, monotonic-safe for durations)."""
    return time.time()


@dataclass
class ChunkRecord:
    """Per-chunk semantic record (adapter layer).

    Chunk identity: (method, case_run_id, sender, msg_id, chunk_id).
    """

    method: str
    case_run_id: str
    sender: int
    msg_id: int
    chunk_id: int
    total_chunks: int
    frame_published: bool
    payload_len: int  # ground-truth payload length in bytes

    # send side
    send_time: float | None = None  # first storage publish op for this chunk
    send_round: int | None = None
    publish_round: int | None = None  # round at which it became visible

    # receive side: per receiver first correct delivery (§7 delivery def)
    delivered: dict[int, bool] = field(default_factory=dict)  # receiver -> bool
    deliver_time: dict[int, float] = field(default_factory=dict)
    deliver_round: dict[int, int] = field(default_factory=dict)

    # decode attempts (candidate bytes kept BEFORE integrity drop)
    # each attempt: dict(receiver, round, time, candidate_hex, crc_ok,
    #                    hamming, comparable, aligned_key)
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chunk_key(self) -> str:
        return f"{self.method}|{self.case_run_id}|{self.sender}|{self.msg_id}|{self.chunk_id}"


@dataclass
class StorageOp:
    """One physical storage operation (backend wrapper layer)."""

    op: str  # file_write | file_read | dir_create | dir_list | cache_set |
    #        cache_probe | cache_delete | promote_move | ...
    bytes_logical: int = 0  # logical bytes touched (0 for pure metadata ops)
    count: int = 1
    detail: str = ""
    wall: float = 0.0
    cpu: float = 0.0


@dataclass
class OverheadStats:
    """Aggregated resource overhead (§8.7), two layers kept separate."""

    tool_calls: int = 0  # tool-call layer: send_payload/receive/... each = 1
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    storage_ops: dict[str, int] = field(default_factory=dict)  # op -> count
    logical_bytes_written: int = 0
    logical_bytes_read: int = 0
    peak_occupancy_objects: int = 0  # files / dirs / set keys
    peak_occupancy_bytes: int = 0
    cpu_time: float = 0.0

    def add_op(self, op: StorageOp) -> None:
        self.storage_ops[op.op] = self.storage_ops.get(op.op, 0) + op.count
        if op.op.endswith("write") or "set" in op.op or "create" in op.op:
            self.logical_bytes_written += op.bytes_logical
        elif op.op.endswith("read") or "probe" in op.op or "list" in op.op:
            self.logical_bytes_read += op.bytes_logical

    def merge(self, other: OverheadStats) -> None:
        self.tool_calls += other.tool_calls
        for k, v in other.tool_calls_by_name.items():
            self.tool_calls_by_name[k] = self.tool_calls_by_name.get(k, 0) + v
        for k, v in other.storage_ops.items():
            self.storage_ops[k] = self.storage_ops.get(k, 0) + v
        self.logical_bytes_written += other.logical_bytes_written
        self.logical_bytes_read += other.logical_bytes_read
        self.peak_occupancy_objects = max(
            self.peak_occupancy_objects, other.peak_occupancy_objects
        )
        self.peak_occupancy_bytes = max(
            self.peak_occupancy_bytes, other.peak_occupancy_bytes
        )
        self.cpu_time += other.cpu_time


class MeasurementLog:
    """Central per-run measurement record.

    Thread-safety: A and B run as two threads in one process (§12); a simple
    RLock guards append-style updates.
    """

    def __init__(self, method: str, case_run_id: str) -> None:
        import threading

        self.method = method
        self.case_run_id = case_run_id
        self._lock = threading.RLock()
        self.chunks: dict[str, ChunkRecord] = {}
        self.events: list[dict[str, Any]] = []  # lifecycle / round events
        self.overhead = OverheadStats()
        self.t_start: float | None = None  # round 1 begin (T_start for G)
        self.t_end: float | None = None  # run termination
        self.rounds_completed: int = 0
        self.submissions: list[dict[str, Any]] = []

    # -- send side -----------------------------------------------------------

    def register_send(
        self,
        sender: int,
        msg_id: int,
        chunk_id: int,
        total_chunks: int,
        frame_published: bool,
        payload_len: int,
    ) -> ChunkRecord:
        key = f"{self.method}|{self.case_run_id}|{sender}|{msg_id}|{chunk_id}"
        with self._lock:
            rec = ChunkRecord(
                method=self.method,
                case_run_id=self.case_run_id,
                sender=sender,
                msg_id=msg_id,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                frame_published=frame_published,
                payload_len=payload_len,
            )
            self.chunks[key] = rec
            return rec

    # -- attempts / delivery -------------------------------------------------

    def record_attempt(
        self, rec: ChunkRecord, attempt: dict[str, Any]
    ) -> None:
        with self._lock:
            rec.attempts.append(attempt)
            rx = attempt.get("receiver")
            if attempt.get("crc_ok") and rx is not None:
                if rx not in rec.delivered:  # first correct delivery
                    rec.delivered[rx] = True
                    rec.deliver_time[rx] = float(attempt.get("time", now()))
                    rec.deliver_round[rx] = int(attempt.get("round", 0))

    # -- overhead / events ----------------------------------------------------

    def record_tool_call(self, name: str) -> None:
        with self._lock:
            self.overhead.tool_calls += 1
            by = self.overhead.tool_calls_by_name
            by[name] = by.get(name, 0) + 1

    def record_storage_op(self, op: StorageOp) -> None:
        with self._lock:
            self.overhead.add_op(op)

    def record_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            event = {"time": now(), **event}
            self.events.append(event)

    # -- occupancy ------------------------------------------------------------

    def update_peak(self, objects: int, nbytes: int) -> None:
        with self._lock:
            self.overhead.peak_occupancy_objects = max(
                self.overhead.peak_occupancy_objects, objects
            )
            self.overhead.peak_occupancy_bytes = max(
                self.overhead.peak_occupancy_bytes, nbytes
            )

    # -- run bookkeeping -------------------------------------------------------

    def mark_run_start(self) -> None:
        with self._lock:
            if self.t_start is None:
                self.t_start = now()

    def mark_run_end(self) -> None:
        with self._lock:
            self.t_end = now()

    # -- serialization -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "method": self.method,
                "case_run_id": self.case_run_id,
                "t_start": self.t_start,
                "t_end": self.t_end,
                "rounds_completed": self.rounds_completed,
                "chunks": {
                    k: {
                        "sender": v.sender,
                        "msg_id": v.msg_id,
                        "chunk_id": v.chunk_id,
                        "total_chunks": v.total_chunks,
                        "frame_published": v.frame_published,
                        "payload_len": v.payload_len,
                        "send_time": v.send_time,
                        "send_round": v.send_round,
                        "publish_round": v.publish_round,
                        "delivered": v.delivered,
                        "deliver_time": v.deliver_time,
                        "deliver_round": v.deliver_round,
                        "attempts": v.attempts,
                    }
                    for k, v in self.chunks.items()
                },
                "overhead": {
                    "tool_calls": self.overhead.tool_calls,
                    "tool_calls_by_name": self.overhead.tool_calls_by_name,
                    "storage_ops": self.overhead.storage_ops,
                    "logical_bytes_written": self.overhead.logical_bytes_written,
                    "logical_bytes_read": self.overhead.logical_bytes_read,
                    "peak_occupancy_objects": self.overhead.peak_occupancy_objects,
                    "peak_occupancy_bytes": self.overhead.peak_occupancy_bytes,
                    "cpu_time": self.overhead.cpu_time,
                },
                "events": self.events,
                "submissions": self.submissions,
            }


def hamming_distance(a: bytes, b: bytes) -> int:
    """Bitwise Hamming distance over the aligned (min-length) prefix."""
    diff = 0
    for x, y in zip(a, b):
        diff += bin(x ^ y).count("1")
    return diff


@dataclass
class ChannelPolicy:
    """Generic isolation-control semantics only (§6, §9-expD).

    Carrier-specific run parameters (e.g. M3's key-set size / TTL) do NOT
    live here: each adapter claims them from the CaseRunConfig it owns.
    """

    mode: str = "shared"  # shared | separated | receiver_readonly
    writable_subset: frozenset[str] | None = None  # allowed object names, or None


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
        config: CaseRunConfig,
    ) -> None:
        import pathlib

        self.root = pathlib.Path(root)
        self.mlog = measurement
        self.config = config
        self.num_agents = config.num_agents
        # generic isolation controls only; carrier-specific config fields are
        # claimed by the subclass constructors, never here
        self.policy = ChannelPolicy(
            mode=config.policy_mode,
            writable_subset=(
                frozenset(config.writable_subset)
                if config.writable_subset is not None
                else None
            ),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_msg_id: dict[int, int] = {i: 0 for i in range(self.num_agents)}
        # send-side ground truth payloads (evaluation only; never shown to
        # agents or monitor W)
        self._truth: dict[tuple[int, int, int], bytes] = {}
        # per-receiver reassembly inbox: rx -> sender -> msg_id -> frames
        self._inbox: dict[int, dict[int, dict[int, list[F.WireFrame]]]] = {
            i: {} for i in range(self.num_agents)
        }
        self._delivered: set[tuple[int, int, int]] = set()
        # per-receiver set of consumed physical object keys
        self._seen: dict[int, set[str]] = {i: set() for i in range(self.num_agents)}
        self._staged: dict[int, list[F.Frame]] = {i: [] for i in range(self.num_agents)}
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


# ---------------------------------------------------- config -> channel bridge

def build_views(adapter: ChannelAdapter, config: CaseRunConfig) -> dict[int, ChannelView]:
    if config.policy_mode == "separated":
        # control (a): each agent gets an independent backend; in this
        # adapter each agent can read only its OWN namespace -> receive()
        # always yields nothing cross-agent (channel cut)
        return {
            i: ChannelView(
                agent_id=i,
                adapter=adapter,
                can_read_senders=frozenset({i}),
                writes_dropped=False,
            )
            for i in range(config.num_agents)
        }
    if config.policy_mode == "receiver_readonly":
        # control (b): B (agent 1) reads everything but its writes drop
        return {
            i: ChannelView(
                agent_id=i,
                adapter=adapter,
                can_read_senders=frozenset(range(config.num_agents)),
                writes_dropped=(i == 1),
            )
            for i in range(config.num_agents)
        }
    return {
        i: ChannelView(
            agent_id=i,
            adapter=adapter,
            can_read_senders=frozenset(range(config.num_agents)),
            writes_dropped=False,
        )
        for i in range(config.num_agents)
    }