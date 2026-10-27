"""Round-trip tests for the frame protocol (§5)."""

import pytest

from network_security_experiment import frame as F


def test_crc16_known_vector():
    # CRC-16/CCITT-FALSE of "123456789" is 0x29B1
    assert F.crc16(b"123456789") == 0x29B1


def test_chunk_reassemble_roundtrip():
    data = b"x" * 200  # 4 chunks: 64+64+64+8
    frames, nxt = F.chunk_bytes(data, 0)
    assert nxt == 1
    assert len(frames) == 4
    assert frames[-1].frame_published is True
    assert all(not f.frame_published for f in frames[:-1])
    assert [f.total_chunks for f in frames] == [4, 4, 4, 4]

    wires = []
    for f in frames:
        raw = f.pack()
        wf = F.WireFrame.parse(raw)
        assert wf is not None and wf.crc_ok
        wires.append(wf)
    assert F.reassemble(wires) == data


def test_pack_layout_matches_header_spec():
    data = b"hello"
    frames, _ = F.chunk_bytes(data, 7)
    f = frames[0]
    raw = f.pack()
    assert len(raw) == F.HEADER_SIZE + 5  # 9B header + payload
    # big-endian fields: msg_id=7, chunk_id=0, total=1, flags=1, crc
    assert raw[0:2] == b"\x00\x07"
    assert raw[2:4] == b"\x00\x00"
    assert raw[4:6] == b"\x00\x01"
    assert raw[6] == 0x01  # frame_published on final chunk


def test_reassemble_rejects_incomplete():
    data = b"y" * 130
    frames, _ = F.chunk_bytes(data, 0)
    wires = [F.WireFrame.parse(f.pack()) for f in frames]
    assert F.reassemble(wires[:1]) is None  # 3 chunks, only 1 present
    assert F.reassemble(wires[:2]) is None
    assert F.reassemble(wires) == data


def test_corrupt_candidate_kept_for_ber():
    data = b"z" * 64
    frames, _ = F.chunk_bytes(data, 0)
    raw = bytearray(frames[0].pack())
    raw[9] ^= 0x01  # flip one payload bit
    wf = F.WireFrame.parse(bytes(raw))
    assert wf is not None
    assert wf.crc_ok is False  # corrupt candidate survives parsing (§8.4)
    assert wf.payload != frames[0].payload


def test_empty_payload_rejected():
    with pytest.raises(ValueError):
        F.chunk_bytes(b"", 0)