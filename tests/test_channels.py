"""Channel round-trip tests: M1 file bodies, M2 directory names, M3 cache keys.

Covers the §6 contracts: unified send/receive, publish-this-round/read-next-
round visibility, decode-candidate retention for BER, and policy controls.
"""

import shutil

import pytest

from network_security_experiment.channels.m1_file import M1FileContentChannel, m1_views
from network_security_experiment.channels.m2_dirname import M2DirNameChannel
from network_security_experiment.channels.m3_cache import M3CacheStateChannel
from network_security_experiment.channels.base import ChannelPolicy
from network_security_experiment.measurement.recorder import MeasurementLog


@pytest.fixture(autouse=True)
def clean_pkg_cache():
    import diskcache

    yield
    diskcache.Cache.disk_cache_dir = None  # no global state, placeholder


def make_channel(cls, tmp_path, policy=None):
    mlog = MeasurementLog(method=cls.method, case_run_id="test-run")
    adapter = cls(
        root=str(tmp_path / cls.method.lower()),
        measurement=mlog,
        policy=policy or ChannelPolicy(),
        num_agents=2,
    )
    return adapter, mlog


def round_trip(adapter, mlog, data: bytes, rounds_needed: int = 2):
    """Send from 0, promote rounds, receive as 1; returns delivered payloads."""
    adapter.set_round(1)
    receipt = adapter.send_payload(0, data)
    adapter.publish_round_end()
    messages = []
    for r in range(2, rounds_needed + 3):
        adapter.set_round(r)
        msgs = adapter.receive(1)
        messages.extend(msgs)
        if msgs:
            break
        adapter.publish_round_end()
    return receipt, messages


def test_m1_roundtrip(tmp_path):
    adapter, mlog = make_channel(M1FileContentChannel, tmp_path)
    receipt, msgs = round_trip(adapter, mlog, b"local max is 827", rounds_needed=2)
    assert receipt["status"] == "staged"
    assert len(msgs) == 1
    assert msgs[0].data == b"local max is 827"
    assert msgs[0].sender == 0
    # measurement: chunk delivered, attempts comparable, hamming 0
    rec = list(mlog.chunks.values())[0]
    assert rec.delivered.get(1) is True
    assert rec.attempts[0]["comparable"] is True
    assert rec.attempts[0]["hamming"] == 0


def test_m1_not_visible_same_round(tmp_path):
    """Writes staged in round r are NOT readable before the round boundary."""
    adapter, mlog = make_channel(M1FileContentChannel, tmp_path)
    adapter.set_round(1)
    adapter.send_payload(0, b"early bird")
    assert adapter.receive(1) == []  # staging only, nothing live yet
    adapter.publish_round_end()
    adapter.set_round(2)
    msgs = adapter.receive(1)
    assert [m.data for m in msgs] == [b"early bird"]


def test_m1_multichunk(tmp_path):
    adapter, mlog = make_channel(M1FileContentChannel, tmp_path)
    data = bytes(range(256)) * 2  # 512 B -> 8 chunks
    _, msgs = round_trip(adapter, mlog, data)
    assert len(msgs) == 1 and msgs[0].data == data


def test_m2_roundtrip(tmp_path):
    adapter, mlog = make_channel(M2DirNameChannel, tmp_path)
    _, msgs = round_trip(adapter, mlog, b"vote: Candidate_B")
    assert [m.data for m in msgs] == [b"vote: Candidate_B"]
    # objects are EMPTY directories
    live = tmp_path / "m2" / "live" / "sender0"
    dirs = [p for p in live.iterdir() if p.is_dir()]
    assert dirs and all(list(p.iterdir()) == [] for p in dirs)
    # names are pure base32 (uppercase alnum)
    assert all(p.name.isalnum() and p.name.isupper() for p in dirs)


def test_m3_roundtrip(tmp_path):
    """M3 with K=128: a 16 B payload needs ceil((4+16)*8 / 112) = 2 rounds."""
    adapter, mlog = make_channel(
        M3CacheStateChannel, tmp_path, ChannelPolicy(key_set_size=128)
    )
    data = b"hi from A"
    receipt = adapter.send_payload(0, data)
    assert receipt["status"] == "staged"
    messages = []
    adapter.set_round(1)
    adapter.publish_round_end()
    for r in range(2, 12):
        adapter.set_round(r)
        messages = adapter.receive(1)
        if messages:
            break
        adapter.publish_round_end()
    assert len(messages) == 1
    assert messages[0].data == data
    rec = list(mlog.chunks.values())[0]
    assert rec.delivered.get(1) is True
    assert rec.publish_round is not None


def test_m3_receiver_skipping_round_loses_segment(tmp_path):
    """Fire-and-forget with multi-segment streams: at K=32 a 9 B payload
    needs ceil(13*8 / 16) = 7 segments (7 rounds).  If the receiver skips
    one read round, that generation's bits are overwritten and never come;
    the misaligned stream cannot CRC-pass."""
    adapter, mlog = make_channel(
        M3CacheStateChannel, tmp_path, ChannelPolicy(key_set_size=32)
    )
    data = b"lost bits"  # 9 B payload -> 13 B stream = 104 bits -> 7 segments
    adapter.set_round(1)
    adapter.send_payload(0, data)
    adapter.publish_round_end()
    # receiver reads every round EXCEPT round 4 (misses generation 4)
    for r in range(2, 10):
        adapter.set_round(r)
        if r != 4:
            adapter.receive(1)
        adapter.publish_round_end()
    adapter.set_round(10)
    msgs = adapter.receive(1)
    assert all(m.data != data for m in msgs)
    rec = list(mlog.chunks.values())[0]
    assert rec.delivered.get(1) is None  # never correctly delivered


def test_m3_receiver_reading_every_round_delivers(tmp_path):
    """Same setup, receiver reads every round: the chunk arrives intact."""
    adapter, mlog = make_channel(
        M3CacheStateChannel, tmp_path, ChannelPolicy(key_set_size=32)
    )
    data = b"lucky bits"
    adapter.set_round(1)
    adapter.send_payload(0, data)
    adapter.publish_round_end()
    got = None
    for r in range(2, 12):
        adapter.set_round(r)
        msgs = adapter.receive(1)
        if msgs:
            got = msgs[0]
            break
        adapter.publish_round_end()
    assert got is not None and got.data == data


def test_policy_receiver_readonly_cuts_only_b2a(tmp_path):
    """Control (b): B's writes drop, but A->B channel stays alive."""
    adapter, mlog = make_channel(M1FileContentChannel, tmp_path)
    views = m1_views(adapter, "receiver_readonly")
    assert views[1].writes_dropped is True
    assert views[0].writes_dropped is False
    adapter.set_round(1)
    adapter.send_payload(0, b"A to B", view=views[0])
    adapter.send_payload(1, b"B to A", view=views[1])
    assert list(adapter._staged[1]) == []  # B's write never staged
    adapter.publish_round_end()
    adapter.set_round(2)
    assert [m.data for m in adapter.receive(1, view=views[1])] == [b"A to B"]
    adapter.publish_round_end()
    adapter.set_round(3)
    assert adapter.receive(0, view=views[0]) == []


def test_policy_writable_subset_blocks_m1(tmp_path):
    """Control (c): shrinking the writable namespace kills the channel."""
    adapter, mlog = make_channel(
        M1FileContentChannel,
        tmp_path,
        ChannelPolicy(writable_subset=frozenset()),  # nothing writable
    )
    adapter.set_round(1)
    receipt = adapter.send_payload(0, b"blocked")
    assert receipt["status"] == "staged"  # fire-and-forget: no error to agent
    adapter.publish_round_end()
    adapter.set_round(2)
    assert adapter.receive(1) == []
    assert mlog.overhead.storage_ops.get("write_blocked_by_policy", 0) >= 1


def test_m3_ttl_passive_expiry(tmp_path):
    """Experiment C: TTL semantics — key in c respects expiry."""
    adapter, mlog = make_channel(
        M3CacheStateChannel,
        tmp_path,
        ChannelPolicy(key_set_size=128, ttl_seconds=0.05),
    )
    adapter.set_round(1)
    adapter.send_payload(0, b"gone soon")
    adapter.publish_round_end()
    import time

    time.sleep(0.1)
    adapter.set_round(2)
    msgs = adapter.receive(1)
    # all keys expired -> header parity over all-zero state passes but bits
    # are zeros -> corrupt/lost stream, or empty: either way NOT delivered
    assert all(m.data != b"gone soon" for m in msgs)