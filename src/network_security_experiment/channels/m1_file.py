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

from ..protocol import frame as F
from ..protocol.frame import M1_BODY_LIMIT
from .base import ChannelAdapter, ChannelView


class M1FileContentChannel(ChannelAdapter):
    method = "M1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slots: dict[int, int] = {i: 0 for i in range(self.num_agents)}
        self._max_live_slot: dict[int, int] = {i: -1 for i in range(self.num_agents)}
        self._live_bytes: dict[int, int] = {i: 0 for i in range(self.num_agents)}

    # ------------------------------------------------------------- send side

    def _stage_frame(self, sender: int, fr: F.Frame) -> None:
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
        with open(path, "wb") as f:
            f.write(body)
        self._op("file_write", nbytes=len(body))
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
                os.replace(path, live_dir / path.name)
                self._op("promote_move", nbytes=size)
                self._live_bytes[sender] += size
                slot = int(path.stem[1:])
                self._max_live_slot[sender] = max(self._max_live_slot[sender], slot)
        shutil.rmtree(stage_dir, ignore_errors=True)
        self.update_peak_occupancy()

    # ---------------------------------------------------------- receive side

    def _read_new_live(self, sender: int, receiver: int) -> list[F.WireFrame]:
        live_dir = self.root / "live" / f"sender{sender}"
        if not live_dir.exists():
            return []
        out: list[F.WireFrame] = []
        seen = self._seen[receiver]
        for path in sorted(live_dir.iterdir()):
            if not path.is_file():
                continue
            name = f"s{sender}/{path.name}"
            if name in seen:
                continue
            seen.add(name)
            with open(path, "rb") as f:
                body = f.read()
            self._op("file_read", nbytes=len(body))
            wf = F.WireFrame.parse(body)
            if wf is not None:
                out.append(wf)
        return out

    def _occupancy(self) -> tuple[int, int]:
        n = sum(s + 1 for s in self._max_live_slot.values())
        return n, sum(self._live_bytes.values())


def m1_views(
    adapter: M1FileContentChannel, policy_mode: str = "shared"
) -> dict[int, ChannelView]:
    """Build per-agent policy-aware views (experiment D controls)."""
    views = {}
    for i in range(adapter.num_agents):
        if policy_mode == "separated":
            can_read = frozenset({i})  # own backend only: no cross visibility
            dropped = False
        elif policy_mode == "receiver_readonly" and i == 1:
            can_read = frozenset({0, 1})  # B still sees A's state
            dropped = True  # but B's writes go nowhere
        else:
            can_read = frozenset(range(adapter.num_agents))
            dropped = False
        views[i] = ChannelView(agent_id=i, adapter=adapter, can_read_senders=can_read, writes_dropped=dropped)
    return views