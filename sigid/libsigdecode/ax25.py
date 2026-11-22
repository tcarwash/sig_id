"""Independent first-party AX.25 UI-frame decoder slice for Bell 202 audio.

Scope: Bell 202 1200 baud AFSK (mark 1200 Hz, space 2200 Hz) mono audio,
receive-only bit recovery, NRZI decoding, HDLC flag sync/de-stuffing, AX.25
address/control/PID parsing, and X.25 FCS validation. Restricted to
unnumbered-information (UI, control 0x03, PID 0xF0) frames carrying arbitrary
information bytes. Written from the TAPR AX.25 v2.2 link-layer
description and the Bell 202 tone/baud parameters; no OnAir/Fldigi/WSJT-X
code or tables are used here.

The AFSK/HDLC helpers below are intentionally private to this file (the
work queue lists ``shared_afsk_hdlc`` as an integrator-owned dependency).
A common-component follow-up is proposed in the evidence dossier.
"""

from __future__ import annotations

import numpy as np

from .core import DecodeWindow, DecodedToken, DecoderSpec, LocalDecodeResult
from .artifacts import DecodedFrame, IntegrityCheck

_MARK_HZ = 1200.0
_SPACE_HZ = 2200.0
_BAUD = 1200.0
_FLAG = 0x7E
_UI_CONTROL = 0x03
_NO_LAYER3_PID = 0xF0
_MIN_SAMPLE_RATE = 8000
_MAX_PROCESS_SECONDS = 8.0
_MIN_WINDOW_SECONDS = 0.125
_RMS_FLOOR = 1e-4
_FRAME_VERSION = "ax25-bell202-ui-v1"


def _x25_fcs(data: bytes | bytearray | memoryview) -> int:
    """X.25 (CRC-16/X-25) frame check sequence over data bytes.

    Init 0xFFFF, reflected polynomial 0x8408, final XOR 0xFFFF. AX.25 sends
    this 16-bit value most-significant bit first on the wire.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc ^ 0xFFFF


def _destuff(gap: list[int]) -> list[int] | None:
    """Remove HDLC zero-bit-stuffing from bits between two flags.

    Returns None when the gap contains a flag/abort pattern (six or more
    consecutive ones), which means it is not a single valid stuffed run.
    """
    out: list[int] = []
    ones = 0
    for bit in gap:
        if bit:
            ones += 1
            if ones >= 6:
                return None
            out.append(1)
        else:
            if ones == 5:
                ones = 0  # Stuffed zero: dropped, not data.
                continue
            if ones >= 6:
                return None
            ones = 0
            out.append(0)
    return out


def _extract_frames(raw_bits: list[int]) -> list[tuple[list[int], int, int]]:
    """Split a raw NRZI-decoded bit list into destuffed candidate frames.

    Returns (frame_bits, opening_flag_bit, closing_flag_end_bit) tuples.
    Frame bits exclude both flags; extents are raw bit indexes for sample
    mapping.
    """
    text = "".join("1" if bit else "0" for bit in raw_bits)
    starts: list[int] = []
    pos = text.find("01111110")
    while pos != -1:
        starts.append(pos)
        pos = text.find("01111110", pos + 1)
    frames: list[tuple[list[int], int, int]] = []
    for first, second in zip(starts, starts[1:]):
        gap = raw_bits[first + 8:second]
        if len(gap) < 8:
            continue
        body = _destuff(gap)
        if body is None or len(body) % 8 != 0 or not body:
            continue
        frames.append((body, first, second + 8))
    return frames


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a bit list (transmission order, LSB first per byte)."""
    out = bytearray(len(bits) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _parse_ui_frame(frame: bytes, received_fcs: int) -> tuple[dict | None, bool]:
    """Parse AX.25 UI-frame bytes (address/control/PID/info/FCS).

    Returns (record, fcs_failed). A structurally invalid frame yields
    (None, False); a structurally sound frame with a bad FCS yields
    (None, True) so the caller can report integrity evidence.
    """
    if len(frame) < 16:  # 2x7 addr + control + PID + >=1 info byte.
        return None, False
    addresses: list[str] = []
    index = 0
    while True:
        if index + 7 > len(frame) or len(addresses) >= 4:
            return None, False
        chunk = frame[index:index + 7]
        last = bool(chunk[6] & 0x01)
        call_chars: list[str] = []
        for byte in chunk[:6]:
            code = (byte >> 1) & 0x7F
            if code == 0x20:
                call_chars.append(" ")
            elif 0x41 <= code <= 0x5A or 0x30 <= code <= 0x39:
                call_chars.append(chr(code))
            else:
                return None, False
        raw_callsign = "".join(call_chars)
        callsign = raw_callsign.rstrip(" ")
        if " " in callsign:
            return None, False
        if not callsign:
            return None, False
        if chunk[6] & 0x60 != 0x60:
            return None, False
        addresses.append(f"{callsign}-{(chunk[6] >> 1) & 0x0F}")
        index += 7
        if last:
            break
    if len(addresses) < 2:
        return None, False
    if index + 3 > len(frame):  # control + PID + >=1 info byte.
        return None, False
    control = frame[index]
    pid = frame[index + 1]
    if control != _UI_CONTROL or pid != _NO_LAYER3_PID:
        return None, False
    info = frame[index + 2:]
    if not info:
        return None, False
    computed = _x25_fcs(frame)
    if computed != received_fcs:
        return None, True
    return {
        "addresses": addresses,
        "control": control,
        "pid": pid,
        "info": bytes(info),
        "fcs_received": received_fcs,
        "fcs_computed": computed,
    }, False


def parse_ax25_ui_frame_bytes(wire_frame: bytes) -> tuple[dict | None, bool | None]:
    """Parse one complete AX.25 UI frame with the project wire-byte order.

    Returns (record, fcs_ok). fcs_ok is None when the bytes are too short or
    structurally unsupported, False for a supported frame with bad FCS, and
    True only for a valid UI/PID-F0 frame with a passing X.25 FCS. This
    byte-level entry point is shared by packet-mode decoders; it does not
    perform AFSK, HDLC, or NRZI processing.
    """
    if not isinstance(wire_frame, bytes) or len(wire_frame) < 2:
        return None, None
    frame = wire_frame[:-2]
    received_fcs = int.from_bytes(wire_frame[-2:], "big")
    record, fcs_failed = _parse_ui_frame(frame, received_fcs)
    if record is not None:
        return record, True
    if fcs_failed:
        return None, False
    return None, None


def _demodulate_tones(
    samples: np.ndarray, rate: int, offset: float, *,
    mark_hz: float = _MARK_HZ, space_hz: float = _SPACE_HZ,
    baud: float = _BAUD,
) -> list[int]:
    """Noncoherent per-bit mark/space decision at one timing offset.

    Mixes with both Bell 202 tones and compares per-bit correlation
    magnitudes using vectorized segment sums. Returns the tone sequence
    (1 for mark, 0 for space).
    """
    bit_len = rate / baud
    count = int((samples.size - offset) // bit_len)
    if count < 2:
        return []
    edges = np.floor(offset + np.arange(count + 1) * bit_len).astype(np.int64)
    edges = np.clip(edges, 0, samples.size)
    while edges.size > 1 and edges[-1] >= samples.size:
        edges = edges[:-1]  # reduceat indices must be < N; drop the partial tail.
        count -= 1
    # Drop degenerate empty cells from float rounding.
    keep = np.ones(count + 1, dtype=bool)
    for i in range(count, 0, -1):
        if edges[i] <= edges[i - 1]:
            keep[i] = False
    edges = edges[keep]
    count = edges.size - 1
    if count < 2:
        return []
    work = samples.astype(np.float32, copy=False)
    n = np.arange(samples.size, dtype=np.float64)
    mix_mark = work * np.exp(-2j * np.pi * mark_hz * n / rate).astype(np.complex64)
    mix_space = work * np.exp(-2j * np.pi * space_hz * n / rate).astype(np.complex64)
    sum_mark = np.add.reduceat(mix_mark, edges)[:-1]
    sum_space = np.add.reduceat(mix_space, edges)[:-1]
    energy_mark = sum_mark.real ** 2 + sum_mark.imag ** 2
    energy_space = sum_space.real ** 2 + sum_space.imag ** 2
    return [1 if m > s else 0 for m, s in zip(energy_mark, energy_space)]


class AX25Decoder:
    """Receive-only two-tone AX.25 UI-frame decoder (experimental)."""

    def __init__(
        self, *, mode_id: str = "ax25", name: str = "AX.25 UI (Bell 202 1200 baud)",
        mark_hz: float = _MARK_HZ, space_hz: float = _SPACE_HZ,
        baud: float = _BAUD, version: str = _FRAME_VERSION,
        modem_description: str = "Bell 202",
    ) -> None:
        if not (0 < mark_hz < space_hz and baud > 0):
            raise ValueError("AX.25 modem requires positive ordered tone frequencies and baud")
        self.mark_hz = float(mark_hz)
        self.space_hz = float(space_hz)
        self.baud = float(baud)
        self.modem_description = modem_description
        self.spec = DecoderSpec(
            id=mode_id, name=name, family="packet", input_kind="mono-audio",
            stage="experimental", version=version,
        )

    def decode(self, window: DecodeWindow) -> LocalDecodeResult:
        candidate_frames: list[DecodedFrame] = []

        def empty(reason: str, evidence: dict | None = None) -> LocalDecodeResult:
            return LocalDecodeResult(
                self.spec.id, self.spec.version, "no_decode",
                evidence=evidence or {}, note=reason, frames=tuple(candidate_frames),
            )

        rate = window.sample_rate
        if rate < _MIN_SAMPLE_RATE:
            return empty(
                f"Bell 202 tones need a sample rate of at least "
                f"{_MIN_SAMPLE_RATE} Hz.",
                {"sample_rate": rate, "required_min_sample_rate": _MIN_SAMPLE_RATE},
            )
        if window.samples.size < int(rate * _MIN_WINDOW_SECONDS):
            return empty(
                "At least 0.125 seconds of audio is needed for one AX.25 frame.",
                {"sample_count": int(window.samples.size), "sample_rate": rate},
            )
        audio = np.asarray(window.samples, dtype=np.float64)
        audio = audio - float(np.mean(audio))
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < _RMS_FLOOR:
            return empty("Audio is silent; no AFSK energy present.", {"rms": rms})
        cap = min(audio.size, int(rate * _MAX_PROCESS_SECONDS))
        truncated = cap < audio.size
        audio = audio[:cap]
        bit_len = rate / self.baud

        candidates = 0
        fcs_failures = 0
        structural_rejects = 0
        for quarter in range(4):
            offset = quarter * bit_len / 4.0
            tones = _demodulate_tones(
                audio, rate, offset, mark_hz=self.mark_hz,
                space_hz=self.space_hz, baud=self.baud,
            )
            if len(tones) < 24:
                continue
            raw = [1 if tones[i] == tones[i - 1] else 0 for i in range(1, len(tones))]
            for body, flag_start, flag_end in _extract_frames(raw):
                data_bits, fcs_bits = body[:-16], body[-16:]
                frame = _bits_to_bytes(data_bits)
                received_fcs = 0
                for bit in fcs_bits:
                    received_fcs = (received_fcs << 1) | bit
                record, fcs_failed = _parse_ui_frame(frame, received_fcs)
                candidates += 1
                if record is not None:
                    # AX.25 information fields may carry binary protocol data.
                    # Keep the exact bytes in frame evidence and use a safe
                    # printable projection for the legacy text field.
                    text = "".join(
                        chr(byte) if 0x20 <= byte <= 0x7E else "�"
                        for byte in record["info"]
                    )
                    start_sample = window.sample_start + int(offset + flag_start * bit_len)
                    end_sample = window.sample_start + int(offset + flag_end * bit_len)
                    end_sample = max(end_sample, start_sample + 1)
                    token = DecodedToken(
                        text, start_sample, end_sample,
                        "AX.25-UI", 1.0,
                    )
                    evidence = {
                        "addresses": record["addresses"],
                        "control": f"0x{record['control']:02X}",
                        "pid": f"0x{record['pid']:02X}",
                        "fcs_ok": True,
                        "fcs_received_hex": f"0x{record['fcs_received']:04X}",
                        "fcs_computed_hex": f"0x{record['fcs_computed']:04X}",
                        "frame_hex": frame.hex() + received_fcs.to_bytes(2, "big").hex(),
                        "info_hex": record["info"].hex(),
                        "frame_sample_start": start_sample,
                        "frame_sample_end": end_sample,
                        "source_window_start_sample": window.sample_start,
                        "source_window_sample_count": int(window.samples.size),
                        "baud": self.baud,
                        "mark_hz": self.mark_hz,
                        "space_hz": self.space_hz,
                        "timing_offset_fraction": quarter / 4.0,
                        "truncated_to_first_8s": truncated,
                        "confidence_kind": "fcs_integrity_check_not_probability",
                    }
                    wire_frame = frame + received_fcs.to_bytes(2, "big")
                    decoded_frame = DecodedFrame(
                        start_sample, end_sample, wire_frame,
                        protocol_hint="AX.25 UI frame",
                        integrity=(IntegrityCheck(
                            "crc", "AX.25 X.25 FCS CRC-16", "passed",
                            note="Computed FCS matches the received 16-bit value.",
                        ),),
                    )
                    return LocalDecodeResult(
                        self.spec.id, self.spec.version, "decoded",
                        text=text, tokens=(token,), evidence=evidence,
                        frames=(decoded_frame,),
                        note=f"Experimental {self.modem_description} AX.25 UI decode; "
                             "integrity-checked by X.25 FCS, not independently verified.",
                    )
                if fcs_failed:
                    fcs_failures += 1
                    start_sample = window.sample_start + int(offset + flag_start * bit_len)
                    end_sample = max(start_sample + 1,
                                     window.sample_start + int(offset + flag_end * bit_len))
                    if len(candidate_frames) < 32:
                        candidate_frames.append(DecodedFrame(
                            start_sample, end_sample,
                            frame + received_fcs.to_bytes(2, "big"),
                            protocol_hint="AX.25 UI candidate",
                            integrity=(IntegrityCheck(
                                "crc", "AX.25 X.25 FCS CRC-16", "failed",
                                note="Received FCS does not match the computed value.",
                            ),),
                        ))
                else:
                    structural_rejects += 1
        return empty(
            "No FCS-valid AX.25 UI frame found; abstaining rather than guessing.",
            {
                "baud": self.baud, "mark_hz": self.mark_hz,
                "space_hz": self.space_hz,
                "rms": round(rms, 6),
                "timing_offsets_tried": 4,
                "inter_flag_candidates": candidates,
                "fcs_failures": fcs_failures,
                "structural_rejects": structural_rejects,
                "truncated_to_first_8s": truncated,
                "source_window_start_sample": window.sample_start,
                "source_window_sample_count": int(window.samples.size),
            },
        )
