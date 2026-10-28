"""M1 — File-Content Blackboard (design doc §6).

A and B exchange data through the *bodies* of files under fixed,
framework-assigned paths.  File names and paths carry no information
(framework-assigned slot numbers); the payload appears only in the file
body, with a fixed-length limit L = 256 B.  The receiver reads only bodies
(never mtime or other attributes).

Layout under root/:
    stage/round{r}/sender{i}/f{slot:06d}.bin   -- staged this round
    live/sender{i}/f{slot:06d}.bin              -- promoted (readable next round)
"""

from __future__ import annotations

import os
import shutil

from . import frame as F
from .frame import M1_BODY_LIMIT
from .base import ChannelAdapter


class M1FileContentChannel(ChannelAdapter):
    method = "M1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slots: dict[int, int] = {i: 0 for i in range(self.num_agents)}
        self._max_live_slot: dict[int, int] = {i: -1 for i in range(self.num_agents)}
        self._live_bytes: dict[int, int] = {i: 0 for i in range(self.num_agents)}

    # ------------------------------------------------------------- send side

    def _stage_frame(self, sender: int, fr: F.Frame) -> bool:
        body = fr.pack()
        if len(body) > M1_BODY_LIMIT:
            raise ValueError(f"M1 body exceeds L={M1_BODY_LIMIT}B")
        slot = self._slots[sender]
        name = f"f{slot:06d}.bin"
        # control (c): only a subset of the shared namespace is writable
        if self.policy.writable_subset is not None and name not in self.policy.writable_subset:
            self._op("write_blocked_by_policy", nbytes=len(body))
            return False
        self._slots[sender] = slot + 1
        stage_dir = self.root / "stage" / f"round{self.current_round}" / f"sender{sender}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / name
        with self._timed_op("file_write", nbytes=len(body)):
            with open(path, "wb") as f:
                f.write(body)
        # §10 audit: W sees the ACTUAL visible file body (M1's carrier is
        # the file content), not the unified tool-call plaintext
        self._audit(
            "file_write", sender, file=name, body_len=len(body),
            body_repr=repr(body[:256]),
        )
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
            if path.is_file():
                size = path.stat().st_size
                with self._timed_op("promote_move", nbytes=size):
                    os.replace(path, live_dir / path.name)
                self._live_bytes[sender] += size
                slot = int(path.stem[1:])
                self._max_live_slot[sender] = max(self._max_live_slot[sender], slot)
                self._audit("promote_move", sender, file=path.name)
        shutil.rmtree(stage_dir, ignore_errors=True)
        self.update_peak_occupancy()

    # ---------------------------------------------------------- receive side

    def _read_new_live(self, sender: int, receiver: int) -> list[F.WireFrame]:
        live_dir = self.root / "live" / f"sender{sender}"
        if not live_dir.exists():
            return []
        out: list[F.WireFrame] = []
        seen = self._seen[receiver]
        with self._timed_op("dir_list", count=1):  # the namespace scan is real work
            entries = [p for p in sorted(live_dir.iterdir()) if p.is_file()]
        for path in entries:
            name = f"s{sender}/{path.name}"
            if name in seen:
                continue
            seen.add(name)
            with self._timed_op("file_read", nbytes=path.stat().st_size):
                with open(path, "rb") as f:
                    body = f.read()
            # §10 audit: which file of whose namespace was read (access
            # association for the extended view)
            self._audit(
                "file_read", receiver, file=path.name, namespace=f"sender{sender}"
            )
            wf = F.WireFrame.parse(body)
            if wf is not None:
                out.append(wf)
        return out

    def _occupancy(self) -> tuple[int, int]:
        n = sum(s + 1 for s in self._max_live_slot.values())
        return n, sum(self._live_bytes.values())