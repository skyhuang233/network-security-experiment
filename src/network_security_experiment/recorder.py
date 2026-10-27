"""Measurement layer records (design doc §7).

Two layers are kept strictly separate:

- Adapter layer (semantic truth): pre-encoding payload bytes on the send side
  (ground truth), decode candidates on the receive side (kept *before*
  integrity checks drop them, for BER), logical msg/chunk ids, logical time.
- Backend wrapper layer (physical overhead): storage operation counts per kind,
  logical bytes read/written, peak occupancy, CPU time.

Truth isolation (§7): send-side ground truth goes ONLY into the evaluation
log; agent B and monitor W never read it, and receive-side records never flow
back to the sender as hidden acks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .frame import PAYLOAD_CHUNK_SIZE


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