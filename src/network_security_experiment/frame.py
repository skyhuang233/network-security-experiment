"""Unified logical frame protocol (design doc §5).

All three channel methods share one logical frame; only the physical
embedding differs (§6):

- M1 (file content): the 9-byte binary frame header + 64-byte payload is
  written as the body of a framework-assigned file (body limit L = 256 B).
- M2 (directory names): the same 73 bytes are base32-encoded (case-safe on
  case-insensitive APFS) into an empty-directory name (<= 240 chars,
  NAME_MAX 255 hard cap).
- M3 (cache key existence): a *reduced* per-publication header occupies a
  fixed small set of reserved keys (msg_id low bits + chunk_id +
  total_chunks low bits + frame_published + parity); the remaining keys carry
  payload bits, 1 bit per key.  The CRC-16 travels in-band as the first two
  bytes of the M3 bit stream.  See m3_cache.py.

Frame fields (all 16-bit except the 1-bit flag, total 65 bit ~ 9 B):
    msg_id        u16  sender-side auto-increment message sequence number
    chunk_id      u16  chunk index within the message
    total_chunks  u16  number of chunks in the message
    frame_published  1 bit  set on the final chunk of the message
    checksum      u16  CRC-16/CCITT-FALSE over the payload bytes

Chunk identity (unique address, §5): (method, case_run_id, sender, msg_id,
chunk_id).  The receiver aligns decode candidates to sender ground truth via
the in-frame (msg_id, chunk_id).

There is NO ack frame and NO protocol-level retransmission: fire-and-forget.
Lost / corrupt chunks flow into the loss counts and BER; the protocol never
retries.  Payload chunk size is 64 B (512 bit) baseline.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# --- constants ---------------------------------------------------------------

PAYLOAD_CHUNK_SIZE = 64  # bytes; baseline chunk (§5)
HEADER_SIZE = 9  # bytes on the wire for M1/M2
M1_BODY_LIMIT = 256  # L = 256 B fixed content limit for M1 file bodies
M2_NAME_CHAR_LIMIT = 240  # <= 240 chars after reserving header/separator room

# M3 per-publication reserved keys (see m3_cache.py):
#   keys 0..7   msg_id low 8 bits        (8 keys)
#   keys 8..13  chunk_id low 6 bits      (6 keys)
#   key  14     frame_published          (1 key)
#   key  15     parity                   (1 key)
#   key  16     publication toggle       (1 key)
#   key  17     activity marker          (1 key)
#   total 18; payload keys start at index 18
M3_RESERVED_KEYS = 18

# --- CRC-16/CCITT-FALSE -------------------------------------------------------

_CRC16_POLY = 0x1021
_CRC16_TABLE: list[int] = []
for _b in range(256):
    _c = _b << 8
    for _ in range(8):
        _c = ((_c << 1) ^ _CRC16_POLY) & 0xFFFF if _c & 0x8000 else (_c << 1) & 0xFFFF
    _CRC16_TABLE.append(_c)


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE over `data`."""
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc & 0xFFFF


# --- frame model ---------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One physical frame: header + payload chunk."""

    msg_id: int
    chunk_id: int
    total_chunks: int
    frame_published: bool  # True on the final chunk of the message
    payload: bytes  # fixed PAYLOAD_CHUNK_SIZE for full chunks, shorter for tail

    def pack(self) -> bytes:
        """Serialize to the 9 B header + payload wire format (M1/M2)."""
        if len(self.payload) > PAYLOAD_CHUNK_SIZE:
            raise ValueError("payload exceeds chunk size")
        flags = 1 if self.frame_published else 0
        header = struct.pack(
            ">HHHBH",
            self.msg_id & 0xFFFF,
            self.chunk_id & 0xFFFF,
            self.total_chunks & 0xFFFF,
            flags,
            crc16(self.payload),
        )
        return header + self.payload

    @classmethod
    def unpack(cls, raw: bytes) -> Frame | None:
        """Parse a wire-format frame.  Returns None if structurally invalid.

        NOTE: the CRC is *not* verified here; callers keep corrupt candidates
        for BER accounting (§8.4) and check the CRC separately.
        """
        if len(raw) < HEADER_SIZE + 1:
            return None
        msg_id, chunk_id, total_chunks, flags, checksum = struct.unpack(
            ">HHHBH", raw[:HEADER_SIZE]
        )
        payload = raw[HEADER_SIZE:]
        if len(payload) > PAYLOAD_CHUNK_SIZE:
            return None
        frame = cls(
            msg_id=msg_id,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            frame_published=bool(flags & 1),
            payload=payload,
        )
        return frame

    def checksum(self) -> int:
        return crc16(self.payload)


@dataclass(frozen=True)
class WireFrame:
    """A frame as recovered from a backend (CRC kept from the wire)."""

    msg_id: int
    chunk_id: int
    total_chunks: int
    frame_published: bool
    stored_checksum: int
    payload: bytes

    @property
    def crc_ok(self) -> bool:
        return crc16(self.payload) == self.stored_checksum

    @classmethod
    def parse(cls, raw: bytes) -> WireFrame | None:
        """Parse raw wire bytes into a candidate frame (no integrity drop)."""
        if len(raw) < HEADER_SIZE + 1:
            return None
        msg_id, chunk_id, total_chunks, flags, checksum = struct.unpack(
            ">HHHBH", raw[:HEADER_SIZE]
        )
        payload = raw[HEADER_SIZE:]
        if len(payload) > PAYLOAD_CHUNK_SIZE:
            return None
        return cls(
            msg_id=msg_id,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            frame_published=bool(flags & 1),
            stored_checksum=checksum,
            payload=payload,
        )


# --- chunking / reassembly -----------------------------------------------------


def chunk_bytes(data: bytes, sender_next_msg_id: int) -> tuple[list[Frame], int]:
    """Split application payload bytes into Frames of one message.

    `sender_next_msg_id` is the message id to use; the caller then bumps the
    sender-side counter.  Returns (frames, next_msg_id_after_this_message).
    """
    if not data:
        raise ValueError("empty payload")
    total = (len(data) + PAYLOAD_CHUNK_SIZE - 1) // PAYLOAD_CHUNK_SIZE
    frames: list[Frame] = []
    for i in range(total):
        piece = data[i * PAYLOAD_CHUNK_SIZE : (i + 1) * PAYLOAD_CHUNK_SIZE]
        frames.append(
            Frame(
                msg_id=sender_next_msg_id,
                chunk_id=i,
                total_chunks=total,
                frame_published=(i == total - 1),
                payload=piece,
            )
        )
    return frames, sender_next_msg_id + 1


def reassemble(frames: list[WireFrame]) -> bytes | None:
    """Reassemble a complete message from CRC-verified frames.

    Returns None unless the frame set is complete (every chunk_id present,
    counts consistent, final flag on the last chunk).
    """
    if not frames:
        return None
    total = frames[0].total_chunks
    if total == 0 or any(f.total_chunks != total for f in frames):
        return None
    by_id: dict[int, WireFrame] = {}
    for f in frames:
        if f.chunk_id >= total:
            return None
        if f.chunk_id in by_id:
            return None  # duplicate
        by_id[f.chunk_id] = f
    if len(by_id) != total:
        return None
    ordered = [by_id[i] for i in range(total)]
    if not ordered[-1].frame_published:
        return None
    return b"".join(f.payload for f in ordered)