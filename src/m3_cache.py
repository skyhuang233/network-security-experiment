"""M3 — Shared-Cache State Signaling (design doc §6, §12).

Backend: DiskCache 5.6.3 (file backend, real on-disk SQLite + files).  A and
B are two threads in one process, each holding an INDEPENDENT `Cache` handle
pointed at the same directory — measured (§2) to be real cross-handle disk
I/O with working expiry semantics, so this does NOT degrade to shared
memory.  No Python objects are ever passed between agents; all information
goes through the storage backend (measurement-layer discipline, §12).

Carrier: key EXISTENCE / absence only (1 bit per key).  Key names carry no
task text and cache VALUES are never read as bodies.

Key set: K keys per sender (`s{sender}_k{index:04d}`), fixed for the run.
Per-publication reserved header keys (18 of them, per sender):
    k00..k07  msg_id, 8 bits              (the reduced header carries only
                                          8 bits: ids >= 256 are REJECTED
                                          at send time.  After 256 messages
                                          the sender refuses further sends
                                          rather than wrapping, so ids are
                                          never aliased onto an old message)
    k08..k13  chunk_id, 6 bits            (ids >= 64 rejected: max message
                                          = 64 chunks x 64 B = 4096 B)
    k14       frame_published (final segment of final chunk only)
    k15       parity over reserved keys k00..k14 (coarse header check)
    k16       publication toggle: flips on EVERY publication, so a receiver
              (including a FRESH one joining later, e.g. the persistence
              test) can tell "new publication since my last read" from the
              physical state alone — no per-receiver bookkeeping
    k17       activity marker: set on the first publication, never cleared —
              distinguishes "never published" (all keys absent) from a real
              publication whose state happens to be all-absent payload keys
Remaining K-18 keys carry payload bits, 1 bit per key, per publication.

PROTOCOL IMPLEMENTATION LIMITS (NOT carrier properties).  The following are
choices of THIS frame protocol implementation, not inherent properties of a
cache-key existence channel, and must not be reported as universal
conclusions about the carrier:
- the reduced 8/6-bit id widths and the 18-key reserved header (a wider
  header or more header keys would address more messages / larger chunks);
- K > 18 being required (an implementation with a leaner header would run
  at smaller K);
- ONE segment of ONE in-flight chunk published per round (P = K-18 payload
  bits per publication; pipelining multiple segments per round is possible
  with a larger state encoding).
These limits are recorded as such wherever they surface (sweep notes,
capacity basis, run events).

Per-chunk bit stream (the 9 B logical frame cannot fit in reserved keys, so
its semantic content travels in-band):
    [total_chunks u8][payload_len u8][crc16 u16 BE][payload <= 64 B]
One *segment* of P = K-16 payload bits is published per round for ONE
in-flight chunk (round-robin among that sender's in-flight chunks; the single
key set expresses exactly one chunk's state per round).  ceil(stream_bits/P)
publications are needed per chunk — "达同样 payload 所需键数/轮数结构性偏多
，是真实结果" (§6).

Expiry discipline (§12, measured on diskcache 5.6.3):
- existence probing uses ONLY `key in c` / `c.get` (both honour expiry);
- `len(c)` / iteration are FORBIDDEN for existence (they still count expired
  entries);
- passive expiry (TTL, `expire=`) => logically unrecoverable, physical rows
  remain on disk (counted under Resource Overhead, not Persistence);
- `c.expire()` performs the "统一资源清理" condition (physical rows removed).
TTL is None in experiment A (persistent, symmetric with M1/M2) and swept
only in experiment C.

Reliability: fire-and-forget.  A receiver that does not read in a round
misses that generation (the next publication overwrites the key state); the
missed bits never arrive — misaligned streams fail the in-band CRC and are
recorded as corrupt candidates (comparable, BER) or lost (incomparable).
"""

from __future__ import annotations

import struct
from collections import deque
from typing import Any

import diskcache

from . import frame as F
from .base import ChannelAdapter


def _bits_of(data: bytes) -> list[int]:
    out = []
    for byte in data:
        for shift in range(7, -1, -1):
            out.append((byte >> shift) & 1)
    return out


def _bytes_of(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


class M3CacheStateChannel(ChannelAdapter):
    method = "M3"

    # The reduced physical header addresses only msg_id < 256 (8 bits).
    # Frames outside the addressable range are REJECTED (_validate_frames).
    # Once 256 messages have been sent, further sends are refused; the
    # counter does not wrap, so chunk records and delivery identity stay put.
    MSG_ID_LIMIT = 0x100

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # M3 claims its own carrier params from the run config (base.py
        # carries only the generic isolation controls)
        self.K = self.config.key_set_size
        self.ttl = self.config.ttl_seconds
        self.cache_size_limit = getattr(
            self.config, "cache_size_limit", 2**30
        )  # diskcache size_limit (eviction pressure is an exp_c condition)
        if self.K <= 18:
            raise ValueError(
                f"M3 needs K > 18 reserved keys (16 header + toggle + "
                f"activity; got K={self.K}); smaller K is structurally "
                "infeasible — record as sweep result"
            )
        self.live_dir = self.root / "live"
        self.live_dir.mkdir(parents=True, exist_ok=True)
        # one independent handle per agent, same directory (real cross-handle I/O)
        self._caches: dict[int, diskcache.Cache] = {}
        # publication generation per sender (0 = nothing published yet);
        # its parity is the physical toggle key k16
        self._gen: dict[int, int] = {i: 0 for i in range(self.num_agents)}
        # last toggle seen by each receiver per sender: (receiver, sender)
        # -> toggle bit or None (None = never read: first read always
        # consumes the current physical state)
        self._rx_toggle: dict[tuple[int, int], int | None] = {}
        # in-flight chunk queues: sender -> deque of (msg_id, chunk_id,
        # total_chunks, frame_published, segments[list[int]], next_seg_idx)
        self._inflight: dict[int, deque] = {i: deque() for i in range(self.num_agents)}
        # per-receiver bit accumulators: rx -> (sender,msg,chunk) -> list[int]
        self._acc: dict[int, dict[tuple[int, int, int], list[int]]] = {
            i: {} for i in range(self.num_agents)
        }
        # round at which a chunk first became live (publish_round correction)
        self._first_pub: dict[tuple[int, int, int], int] = {}

    # ---------------------------------------------------------------- keys

    def _cache(self, agent_id: int) -> diskcache.Cache:
        if agent_id not in self._caches:
            self._caches[agent_id] = diskcache.Cache(
                str(self.live_dir), size_limit=self.cache_size_limit
            )
        return self._caches[agent_id]

    def _key(self, sender: int, index: int) -> str:
        return f"s{sender}_k{index:04d}"

    def _payload_indices(self) -> list[int]:
        if self.policy.writable_subset is not None:
            return [
                i
                for i in range(18, self.K)
                if str(i) in self.policy.writable_subset
            ]
        return list(range(18, self.K))

    # ------------------------------------------------------------- send side

    def _validate_frames(self, frames: list[F.Frame]) -> None:
        """Reject frames the reduced physical header cannot address.

        The M3 header carries msg_id in 8 bits, chunk_id in 6 bits and
        total_chunks in 8 bits.  A logical frame outside these widths would
        be silently ALIASED onto an old message by the bit masks, so it is
        rejected outright (ValueError -> tool error to the agent).  Max
        addressable message: 64 chunks x 64 B = 4096 B; max 256 messages per
        sender, after which further sends are refused (no id reuse)."""
        for fr in frames:
            if fr.msg_id > 0xFF or fr.chunk_id > 0x3F or fr.total_chunks > 0xFF:
                raise ValueError(
                    f"M3 physical header cannot address this frame without "
                    f"aliasing (msg_id<=255, chunk_id<=63, "
                    f"total_chunks<=255 required; got msg_id={fr.msg_id}, "
                    f"chunk_id={fr.chunk_id}, total_chunks={fr.total_chunks}); "
                    "send smaller/fewer messages (max 64 chunks x 64 B)"
                )

    def _stage_frame(self, sender: int, fr: F.Frame) -> bool:
        # masks are identity here: _validate_frames already rejected anything
        # outside the physical widths
        stream = struct.pack(">BBH", fr.total_chunks & 0xFF, len(fr.payload),
                             F.crc16(fr.payload)) + fr.payload
        bits = _bits_of(stream)
        idx = self._payload_indices()
        if not idx:
            self._op("write_blocked_by_policy", nbytes=len(stream))
            return False
        P = len(idx)
        segments = [bits[i : i + P] for i in range(0, len(bits), P)] or [[]]
        self._inflight[sender].append(
            {
                "msg_id": fr.msg_id,
                "chunk_id": fr.chunk_id,
                "total_chunks": fr.total_chunks,
                "frame_published": fr.frame_published,
                "segments": segments,
                "next": 0,
            }
        )
        return True

    # ------------------------------------------------------- round boundary

    def _promote(self, sender: int) -> None:
        """Publish ONE segment of ONE in-flight chunk (round-robin)."""
        if not self._inflight[sender]:
            return
        # rotate: pop front, publish its next segment, requeue if incomplete
        for _ in range(len(self._inflight[sender])):
            chunk = self._inflight[sender].popleft()
            if chunk["next"] >= len(chunk["segments"]):
                continue  # fully published already
            seg = chunk["segments"][chunk["next"]]
            chunk["next"] += 1
            if chunk["next"] < len(chunk["segments"]):
                self._inflight[sender].append(chunk)
            self._publish_segment(sender, chunk, seg)
            key_id = (sender, chunk["msg_id"], chunk["chunk_id"])
            if key_id not in self._first_pub:
                self._first_pub[key_id] = self.current_round
            return

    def _publish_segment(self, sender: int, chunk: dict, seg: list[int]) -> None:
        cache = self._cache(sender)
        idx = self._payload_indices()
        # reserved keys: msg_id low 8, chunk_id low 6, published, parity,
        # then the publication toggle (k16).  The masks are IDENTITY:
        # _validate_frames rejected anything outside the physical widths at
        # stage time, so no id is ever silently truncated here.
        mid = chunk["msg_id"] & 0xFF
        cid = chunk["chunk_id"] & 0x3F
        pub = 1 if (chunk["frame_published"] and chunk["next"] == len(chunk["segments"])) else 0
        header_bits = [int((mid >> (7 - i)) & 1) for i in range(8)] + [
            int((cid >> (5 - i)) & 1) for i in range(6)
        ] + [pub]
        parity = sum(header_bits) & 1
        header_bits.append(parity)
        self._gen[sender] += 1
        toggle = self._gen[sender] & 1
        header_bits.append(toggle)  # k16
        header_bits.append(1)  # k17 activity marker (set forever)
        changes: list[dict[str, Any]] = []  # §10 audit trail (per key op)
        # payload keys: segment bit j lands on allowed key idx[j]
        for j, k in enumerate(idx):
            want = 1 if j < len(seg) and seg[j] else 0
            key = self._key(sender, k)
            with self._timed_op("cache_probe_publish", count=1):
                present = key in cache  # existence probe ONLY (§12)
            if want and not present:
                with self._timed_op("cache_set", count=1):
                    cache.set(key, b"1", expire=self.ttl)
                changes.append({"key": key, "action": "set"})
            elif not want and present:
                with self._timed_op("cache_delete", count=1):
                    cache.delete(key)
                changes.append({"key": key, "action": "delete"})
        # reserved keys are protocol-owned and always writable
        for i, bit in enumerate(header_bits):
            key = self._key(sender, i)
            with self._timed_op("cache_probe_publish", count=1):
                present = key in cache
            if bit and not present:
                with self._timed_op("cache_set", count=1):
                    cache.set(key, b"h", expire=self.ttl)
                changes.append({"key": key, "action": "set"})
            elif not bit and present:
                with self._timed_op("cache_delete", count=1):
                    cache.delete(key)
                changes.append({"key": key, "action": "delete"})
        # §10 audit: actual cache-key state operations (M3's carrier); the
        # content view gets the op summary only, keys go to the extended view
        self._audit(
            "cache_state_change", sender,
            changes=changes[:256], generation=self._gen[sender],
        )
        self.update_peak_occupancy()

    def _post_promote(self) -> None:
        """Correct publish_round: M3 chunks become live at first segment,
        not at stage time (the base class stamps stage-time rounds)."""
        for (sender, mid, cid), rnd in self._first_pub.items():
            rec = self.mlog.chunks.get(self._chunk_key(sender, mid, cid))
            if rec is not None:
                rec.publish_round = rnd

    # ---------------------------------------------------------- receive side

    def _read_new_live(self, sender: int, receiver: int) -> list[F.WireFrame]:
        cache = self._cache(receiver)  # receiver uses its own handle
        # existence probes ONLY (never len()/iteration, §12)
        with self._timed_op("cache_probe", count=self.K):
            state = [1 if self._key(sender, k) in cache else 0 for k in range(self.K)]
        toggle = state[16]
        if state[17] == 0:
            # nothing was EVER published by this sender: the all-absent
            # state is not a publication; never consume it
            self._audit(
                "cache_probe", receiver, namespace=f"sender{sender}",
                keys_probed=self.K, new_publication=False,
            )
            return []
        rkey = (receiver, sender)
        last = self._rx_toggle.get(rkey)
        if last is not None and toggle == last:
            # §10 audit: every executed probe is recorded, even when no new
            # publication is visible
            self._audit(
                "cache_probe", receiver, namespace=f"sender{sender}",
                keys_probed=self.K, new_publication=False,
            )
            return []  # no new publication since this receiver last read
        self._rx_toggle[rkey] = toggle
        self._audit(
            "cache_probe", receiver, namespace=f"sender{sender}",
            keys_probed=self.K, new_publication=True,
        )
        header_bits = state[:16]
        mid = 0
        for b in header_bits[:8]:
            mid = (mid << 1) | b
        cid = 0
        for b in header_bits[8:14]:
            cid = (cid << 1) | b
        pub = header_bits[14]
        parity = sum(header_bits[:15]) & 1
        if parity != header_bits[15]:
            self._op("header_parity_fail", count=1)
            return []  # header untrusted: this generation's bits are lost
        idx = self._payload_indices()
        payload_bits = [state[k] for k in idx]
        key = (sender, mid, cid)
        acc = self._acc[receiver].setdefault(key, [])
        acc.extend(payload_bits)
        status, wf = self._try_parse(mid, cid, pub, acc)
        if status == "wait":
            return []
        # frame completed (crc-checked later) or corrupt stream: consume acc
        del self._acc[receiver][key]
        if wf is None:
            self._op("stream_corrupt_unalignable", count=1)
            return []  # incomparable loss (§8.4)
        return [wf]

    def _try_parse(
        self, mid: int, cid: int, pub: int, acc: list[int]
    ) -> tuple[str, F.WireFrame | None]:
        """("wait", None) while incomplete; ("frame", wf) or ("corrupt", None)."""
        if len(acc) < 32:
            return "wait", None
        head = _bytes_of(acc[:32])
        total_chunks, plen, crc = struct.unpack(">BBH", head)
        if plen == 0 or plen > F.PAYLOAD_CHUNK_SIZE:
            return "corrupt", None  # misaligned stream: incomparable loss
        needed = 32 + 8 * plen
        if len(acc) < needed:
            return "wait", None
        stream = _bytes_of(acc[:needed])
        payload = stream[4:]
        return (
            "frame",
            F.WireFrame(
                msg_id=mid,
                chunk_id=cid,
                total_chunks=total_chunks,
                frame_published=bool(pub),
                stored_checksum=crc,
                payload=payload,
            ),
        )


    def _occupancy(self) -> tuple[int, int | None]:
        """Live SET keys (existence probes only, §12) + ACTUAL physical
        storage footprint of the cache directory (SQLite + value files).

        The byte total is the real on-disk occupancy of the backend, not a
        logical payload count (§8.7)."""
        n = 0
        cache = self._cache(0)
        for sender in range(self.num_agents):
            for k in range(self.K):
                if self._key(sender, k) in cache:
                    n += 1
        nbytes = sum(
            p.stat().st_size for p in self.live_dir.rglob("*") if p.is_file()
        )
        return n, nbytes