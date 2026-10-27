"""M2 — Directory-Name-Based channel (design doc §6).

A and B may only create / rename / list EMPTY directories inside a shared
namespace; no file bodies are used.  The payload is base32-encoded (5 bit per
char, case-safe on case-insensitive APFS) into the directory *name*; the name
length is capped at 240 chars (NAME_MAX 255 hard cap with room to spare).
The receiver lists the namespace, sorts by the in-name chunk identity, and
reconstructs — never timestamps.

Name = base32(frame.pack()) where frame.pack() is the 9 B header + payload
(<= 73 B -> <= 120 chars, well under 240).  Names carry the full logical
frame: msg_id, chunk_id, total_chunks, frame_published, CRC-16.

Layout under root/:
    stage/round{r}/sender{i}/<b32name>/    -- staged empty dirs this round
    live/sender{i}/<b32name>/              -- promoted, readable next round
"""

from __future__ import annotations

import base64
import binascii
import shutil

from . import frame as F
from .frame import M2_NAME_CHAR_LIMIT
from .base import ChannelAdapter


def encode_name(raw: bytes) -> str:
    """base32 without trailing '=' padding (pure uppercase alnum, ≤ 240 chars)."""
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def decode_name(name: str) -> bytes | None:
    try:
        return base64.b32decode(name.upper() + "=" * ((8 - len(name) % 8) % 8))
    except (binascii.Error, ValueError):
        return None


class M2DirNameChannel(ChannelAdapter):
    method = "M2"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._live_dirs: dict[int, int] = {i: 0 for i in range(self.num_agents)}

    # ------------------------------------------------------------- send side

    def _stage_frame(self, sender: int, fr: F.Frame) -> bool:
        name = encode_name(fr.pack())
        if len(name) > M2_NAME_CHAR_LIMIT:
            raise ValueError(f"M2 name exceeds {M2_NAME_CHAR_LIMIT} chars")
        # control (c): restrict the writable subset of the namespace
        if (
            self.policy.writable_subset is not None
            and name[:1] not in self.policy.writable_subset
        ):
            self._op("dir_create_blocked", nbytes=len(name))
            return False
        stage_dir = self.root / "stage" / f"round{self.current_round}" / f"sender{sender}"
        path = stage_dir / name
        if path.exists():
            return True  # exact resend: idempotent (same frame content)
        path.mkdir(parents=True, exist_ok=False)  # EMPTY directory only
        self._op("dir_create", nbytes=len(name))
        return True

    # ------------------------------------------------------- round boundary

    def _promote(self, sender: int) -> None:
        stage_dir = (
            self.root / "stage" / f"round{self.current_round}" / f"sender{sender}"
        )
        if not stage_dir.exists():
            return
        live_dir = self.root / "live" / f"sender{sender}"
        live_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage_dir.iterdir()):
            if path.is_dir():
                target = live_dir / path.name
                if not target.exists():
                    path.rename(target)  # rename = the allowed M2 mutation
                    self._op("promote_rename")
                    self._live_dirs[sender] += 1
        shutil.rmtree(stage_dir, ignore_errors=True)
        self.update_peak_occupancy()

    # ---------------------------------------------------------- receive side

    def _read_new_live(self, sender: int, receiver: int) -> list[F.WireFrame]:
        live_dir = self.root / "live" / f"sender{sender}"
        if not live_dir.exists():
            return []
        out: list[F.WireFrame] = []
        seen = self._seen[receiver]
        fresh = [
            path
            for path in sorted(live_dir.iterdir())
            if path.is_dir() and f"s{sender}/{path.name}" not in seen
        ]
        if not fresh:
            return out
        self._op("dir_list", count=1)  # one list operation per namespace scan
        for path in fresh:
            seen.add(f"s{sender}/{path.name}")
            raw = decode_name(path.name)
            if raw is None:
                continue  # non-protocol name: not a candidate frame
            wf = F.WireFrame.parse(raw)
            if wf is not None:
                out.append(wf)
        return out

    def _occupancy(self) -> tuple[int, int]:
        n = sum(self._live_dirs.values())
        return n, 0