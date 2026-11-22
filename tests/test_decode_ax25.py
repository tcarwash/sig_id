"""Independent Bell 202 / AX.25 UI-frame vectors for the in-project decoder.

The fixture encoder below is written directly from the TAPR AX.25 v2.2
framing description (LSB-first bytes, HDLC flags/bit-stuffing, NRZI tone
mapping, X.25 FCS) and shares no code with sigid/libsigdecode/ax25.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigid.libsigdecode import DecodeWindow
from sigid.libsigdecode.ax25 import AX25Decoder

MARK_HZ = 1200.0
SPACE_HZ = 2200.0
BAUD = 1200.0


def _fixture_fcs(data: bytes) -> int:
    """Test-side X.25 FCS (own implementation, mirrors the standard)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc >> 1) ^ 0x8408) if crc & 1 else (crc >> 1)
    return crc ^ 0xFFFF


def _address_bytes(callsign: str, ssid: int, last: bool) -> bytes:
    padded = callsign.ljust(6)[:6]  # Case preserved: lowercase is the malformed test.
    out = bytearray()
    for char in padded:
        out.append((ord(char) << 1) & 0xFF)
    out.append(0x60 | ((ssid & 0xF) << 1) | (0x01 if last else 0x00))
    return bytes(out)


def _build_frame_bytes(
    dest: str, src: str, info: bytes, dest_ssid: int = 0, src_ssid: int = 0,
    corrupt_info: bool = False,
) -> bytes:
    header = (
        _address_bytes(dest, dest_ssid, False)
        + _address_bytes(src, src_ssid, True)
        + bytes([0x03, 0xF0])
    )
    body = header + info
    fcs = _fixture_fcs(body)
    if corrupt_info:
        # Flip the LSB of the first info byte but keep the original FCS:
        # structure still parses, integrity must fail.
        info = bytes([info[0] ^ 0x01]) + info[1:]
        body = header + info
    return body + fcs.to_bytes(2, "big")


def _to_tone_bits(frame: bytes, corrupt_bit: int | None = None) -> list[int]:
    """Frame bytes -> stuffed raw channel bits (TX order, pre-NRZI)."""
    bits: list[int] = []
    data, fcs = frame[:-2], int.from_bytes(frame[-2:], "big")
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    bits.extend((fcs >> i) & 1 for i in range(15, -1, -1))
    stuffed: list[int] = []
    ones = 0
    for bit in bits:
        stuffed.append(bit)
        if bit:
            ones += 1
            if ones == 5:
                stuffed.append(0)
                ones = 0
        else:
            ones = 0
    if corrupt_bit is not None:
        stuffed[corrupt_bit] ^= 1
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    return flag + flag + stuffed + flag


def _afsk_waveform(
    channel_bits: list[int],
    rate: int = 12000,
    mark_hz: float = MARK_HZ,
    space_hz: float = SPACE_HZ,
    baud: float = BAUD,
    amplitude: float = 0.5,
    noise_std: float = 0.0,
    lead_seconds: float = 0.05,
) -> np.ndarray:
    """Phase-continuous 2-FSK modulator; NRZI: 0 flips the tone, 1 holds it."""
    per = rate / baud
    total = int(lead_seconds * rate) + int(len(channel_bits) * per) + int(0.05 * rate)
    out = np.zeros(total, dtype=np.float64)
    phase = 0.0
    mark = True
    cursor = int(lead_seconds * rate)
    for bit in channel_bits:
        if bit == 0:
            mark = not mark
        freq = mark_hz if mark else space_hz
        n = int(round((cursor + per))) - cursor
        t = np.arange(n, dtype=np.float64)
        out[cursor:cursor + n] = amplitude * np.sin(phase + 2 * np.pi * freq * t / rate)
        phase = (phase + 2 * np.pi * freq * n / rate) % (2 * np.pi)
        cursor += n
    if noise_std:
        out = out + np.random.default_rng(20261122).normal(0, noise_std, out.shape)
    return out


def _valid_audio(text: str, rate: int, **kwargs) -> np.ndarray:
    frame = _build_frame_bytes("APRS", "N0CALL", text.encode("ascii"))
    return _afsk_waveform(_to_tone_bits(frame), rate=rate, **kwargs)


@pytest.mark.parametrize("rate,text", [
    (12000, "HELLO WORLD"),
    (48000, ">TEST:123 de N0CALL"),
])
def test_valid_ui_frame_text_and_fcs(rate, text):
    audio = _valid_audio(text, rate, noise_std=0.004)
    result = AX25Decoder().decode(DecodeWindow(audio, rate, sample_start=5000))
    assert result.status == "decoded", result.to_dict()
    assert result.text == text
    assert result.evidence["fcs_ok"] is True
    assert result.evidence["addresses"] == ["APRS-0", "N0CALL-0"]
    assert result.evidence["control"] == "0x03"
    assert result.evidence["pid"] == "0xF0"
    assert bytes.fromhex(result.evidence["info_hex"]).decode("ascii") == text
    assert len(result.tokens) == 1
    token = result.tokens[0]
    assert token.start_sample >= 5000
    assert token.end_sample > token.start_sample
    assert result.to_dict()["source"] == "sigid-independent-implementation"


def test_valid_ui_frame_preserves_binary_information_without_control_text():
    info = b"\x01\x80binary\x7f"
    frame = _build_frame_bytes("APRS", "N0CALL", info)
    audio = _afsk_waveform(_to_tone_bits(frame), rate=12000)
    result = AX25Decoder().decode(DecodeWindow(audio, 12000))
    assert result.status == "decoded", result.to_dict()
    assert result.evidence["info_hex"] == info.hex()
    assert result.frames[0].payload[-len(info) - 2:-2] == info
    assert all(char == "�" or char.isprintable() for char in result.text)


def test_corrupted_fcs_abstains():
    frame = _build_frame_bytes("APRS", "N0CALL", b"HELLO WORLD", corrupt_info=True)
    audio = _afsk_waveform(_to_tone_bits(frame), rate=12000)
    result = AX25Decoder().decode(DecodeWindow(audio, 12000))
    assert result.status == "no_decode", result.to_dict()
    assert result.text == ""
    assert result.evidence["fcs_failures"] >= 1


def test_silence_and_noise_abstain():
    silence = np.zeros(12000 * 2, dtype=np.float64)
    result = AX25Decoder().decode(DecodeWindow(silence, 12000))
    assert result.status == "no_decode"
    assert result.text == ""
    noise = np.random.default_rng(7).normal(0, 0.05, 12000 * 2)
    result = AX25Decoder().decode(DecodeWindow(noise, 12000))
    assert result.status == "no_decode", result.to_dict()
    assert result.text == ""


def test_malformed_address_abstains():
    # Lowercase callsign bytes violate the AX.25 address alphabet even
    # though the FCS below is computed over exactly these bytes.
    body = (
        _address_bytes("aprs", 0, False)
        + _address_bytes("n0call", 0, True)
        + bytes([0x03, 0xF0])
        + b"HELLO"
    )
    fcs = _fixture_fcs(body)
    frame = body + fcs.to_bytes(2, "big")
    audio = _afsk_waveform(_to_tone_bits(frame), rate=12000)
    result = AX25Decoder().decode(DecodeWindow(audio, 12000))
    assert result.status == "no_decode", result.to_dict()
    assert result.text == ""
    assert result.evidence["fcs_failures"] == 0


def test_x25_check_vector_and_fcs_msb_first_wire_order():
    assert _fixture_fcs(b"123456789") == 0x906E
    frame = b"123456789" + _fixture_fcs(b"123456789").to_bytes(2, "big")
    assert _to_tone_bits(frame)[16:16 + 72] == [
        (byte >> bit) & 1 for byte in b"123456789" for bit in range(8)
    ]
    assert _to_tone_bits(frame)[16 + 72:16 + 72 + 16] == [
        (0x906E >> bit) & 1 for bit in range(15, -1, -1)
    ]
    frame = _build_frame_bytes("APRS", "N0CALL", b"FCS ORDER")
    audio = _afsk_waveform(_to_tone_bits(frame), rate=12000)
    result = AX25Decoder().decode(DecodeWindow(audio, 12000))
    assert result.status == "decoded", result.to_dict()
    assert result.text == "FCS ORDER"


def test_address_limit_and_padding_validation():
    base = _address_bytes("APRS", 0, False) + _address_bytes("N0CALL", 0, False)
    repeaters = b"".join(_address_bytes(f"R{i}", 0, False) for i in range(3))
    body = base + repeaters + _address_bytes("END", 0, True) + bytes([0x03, 0xF0]) + b"HI"
    frame = body + _fixture_fcs(body).to_bytes(2, "big")
    result = AX25Decoder().decode(DecodeWindow(_afsk_waveform(_to_tone_bits(frame)), 12000))
    assert result.status == "no_decode"

    # Callsign padding is trailing only; an internal space is invalid.
    malformed = bytearray(_address_bytes("APRS", 0, False))
    malformed[2] = (ord(" ") << 1)
    body = bytes(malformed) + _address_bytes("N0CALL", 0, True) + bytes([0x03, 0xF0]) + b"HI"
    frame = body + _fixture_fcs(body).to_bytes(2, "big")
    result = AX25Decoder().decode(DecodeWindow(_afsk_waveform(_to_tone_bits(frame)), 12000))
    assert result.status == "no_decode"


def test_wrong_tone_and_speed_abstain():
    frame = _build_frame_bytes("APRS", "N0CALL", b"HELLO WORLD")
    bits = _to_tone_bits(frame)
    off_tone = _afsk_waveform(bits, rate=12000, mark_hz=1700.0, space_hz=2500.0)
    result = AX25Decoder().decode(DecodeWindow(off_tone, 12000))
    assert result.status == "no_decode", result.to_dict()
    assert result.text == ""
    off_speed = _afsk_waveform(bits, rate=12000, baud=2400.0)
    result = AX25Decoder().decode(DecodeWindow(off_speed, 12000))
    assert result.status == "no_decode", result.to_dict()
    assert result.text == ""


def test_decoder_rejects_unsupported_and_short_input():
    decoder = AX25Decoder()
    assert decoder.spec.id == "ax25"
    assert decoder.spec.stage == "experimental"
    assert decoder.spec.input_kind == "mono-audio"
    short = np.zeros(100, dtype=np.float64)
    result = decoder.decode(DecodeWindow(short, 12000))
    assert result.status == "no_decode"
    assert result.text == ""
    slow = _valid_audio("HI", 12000)
    result = decoder.decode(DecodeWindow(slow[:6000], 4000))
    assert result.status == "no_decode"
