"""Curated, locally authored signal field-guide content."""
from __future__ import annotations

import json
from importlib.resources import files
import re
from typing import Any

FAMILY_GUIDANCE: dict[str, dict[str, str]] = {
    "Morse": {"render": "morse", "look_for": "Short and long keyed bursts with clear gaps; timing is the main clue.", "pitfall": "A narrow carrier or impulsive interference can resemble a short Morse element."},
    "PSK": {"render": "psk", "look_for": "A narrow, nearly constant-width trace; information is carried by phase changes, which an ordinary waterfall may show only weakly.", "pitfall": "Do not infer PSK from a thin trace alone; inspect decoded symbols or compare a phase-aware view."},
    "FSK": {"render": "fsk", "look_for": "Energy hops between two or more stable audio tones; transitions form parallel tracks.", "pitfall": "A second nearby carrier can look like a mark/space pair. Check whether energy alternates in time."},
    "MFSK": {"render": "mfsk", "look_for": "A sequence of discrete tone positions, often forming a stepped or scattered horizontal pattern.", "pitfall": "Multiple simultaneous carriers are different from one tone moving among bins."},
    "OFDM": {"render": "ofdm", "look_for": "Many evenly spaced subcarriers active together, often in repeated blocks.", "pitfall": "A comb of unrelated carriers can mimic a multicarrier mode; look for coordinated timing."},
    "Hell": {"render": "hell", "look_for": "Text-like pixel strokes or columns in a waterfall; some variants are keyed tones and others are multitone.", "pitfall": "The visual glyph pattern can be hard to distinguish from other bursty digital signals without timing and width context."},
    "Image": {"render": "image", "look_for": "A steady sweep or repeated line structure whose content changes gradually across the passband.", "pitfall": "A waterfall timebase or tuning offset can distort apparent image width and line rate."},
    "Digital": {"render": "digital", "look_for": "Structured bursts or tone patterns; use occupied bandwidth, cadence, and modulation clues to narrow the family.", "pitfall": "The Digital family is broad. Avoid treating a family label as a protocol identification."},
    "Voice": {"render": "voice", "look_for": "A changing broadband envelope with speech-like syllabic modulation.", "pitfall": "Noise, music, and data bursts can share a broad appearance; listen or use a decoder before labeling."},
    "Time signal": {"render": "morse", "look_for": "Repeated timing marks or coded pulses on a stable carrier.", "pitfall": "A single short recording may not include enough of the repeating schedule."},
    "Test/measurement": {"render": "fsk", "look_for": "Stable tones, sweeps, or repeated calibration patterns.", "pitfall": "A test signal can look like a communications mode but may have no message framing."},
}


# Original identification notes are locally authored. Curated SigIDWiki image
# references are cached separately in user data and retain per-file provenance.
COMMON_SIGNALS: list[dict[str, Any]] = [
    {"id":"ft8","name":"FT8","family":"MFSK","kind":"mfsk","scope":"HF · amateur","modulation":"8-tone FSK","look_for":"Short, synchronized transmissions on a narrow group of discrete tone positions, repeated in a regular cycle.","pitfall":"A single short burst is weak evidence. Compare its cadence and narrow tone spacing with other synchronized digital modes.","context":"Weak-signal amateur contacts; messages are highly structured and brief.","wiki":"https://sigidwiki.com/wiki/FT8","frequency_note":"Common on HF amateur bands; consult a current band plan for operating frequencies.","lookup":{"bandwidth_hz":[35,100],"tone_count":[6,10]},"overlay_scale":{"bandwidth_hz":50,"duration_seconds":12.64}},
    {"id":"wspr","name":"WSPR","family":"MFSK","kind":"mfsk","scope":"HF · amateur","modulation":"4-tone FSK","look_for":"Very narrow, slowly changing tone positions in long, regularly timed transmissions.","pitfall":"Its long transmission duration matters: a short crop can hide the identifying cadence.","context":"Weak-signal propagation beacons rather than conversational text.","wiki":"https://www.sigidwiki.com/wiki/WSPR","frequency_note":"Common on HF amateur bands; consult a current band plan for operating frequencies.","lookup":{"bandwidth_hz":[3,25],"tone_count":[3,5]}},
    {"id":"aprs","name":"APRS / packet radio","family":"FSK","kind":"fsk","scope":"VHF · amateur","modulation":"1200-baud AFSK is common","look_for":"Brief packet bursts with two audio tones and framing-like repetition; often heard as a short chirp or raspy burst on FM voice channels.","pitfall":"The audio tones are an audio-frequency modem over an RF FM channel; do not mistake the audio tones for RF carrier spacing.","context":"Position, telemetry, and short messages on amateur packet networks.","wiki":"https://www.sigidwiki.com/wiki/APRS","frequency_note":"Often associated with 2 m amateur FM; regional channel plans vary.","lookup":{"bandwidth_hz":[600,1500],"tone_count":[2,2]}},
    {"id":"acars","name":"ACARS","family":"FSK","kind":"fsk","scope":"VHF · aviation","modulation":"MSK / AFSK variants","look_for":"Short narrowband aviation data bursts, often appearing as abrupt, compact blocks with crisp edges.","pitfall":"Several aviation data links share the band. A waterfall alone may not distinguish ACARS from other burst systems.","context":"Aircraft-to-ground digital messaging.","wiki":"https://www.sigidwiki.com/wiki/ACARS","frequency_note":"VHF aeronautical allocation; frequency depends on region and service.","lookup":{"bandwidth_hz":[700,1800],"tone_count":[2,2]}},
    {"id":"ads-b","name":"ADS-B","family":"Pulse / data","kind":"ppm","scope":"1090 MHz · aviation","modulation":"Pulse-position modulation","look_for":"Very brief repeated pulse trains in an RF/spectrum view; this is not an audio-passband tone mode.","pitfall":"A normal HF audio recording cannot show the 1090 MHz RF pulses directly; use a suitable RF receiver and decoder.","context":"Broadcast aircraft identity and position reports.","wiki":"https://www.sigidwiki.com/wiki/ADS-B","frequency_note":"1090 MHz (and 978 MHz UAT in the United States)."},
    {"id":"ais","name":"AIS","family":"FSK","kind":"gmsk","scope":"VHF maritime","modulation":"GMSK","look_for":"Short, tightly shaped digital bursts on maritime VHF channels, often in pairs of nearby channels.","pitfall":"It requires VHF reception; the schematic does not imply it is visible in an HF audio passband.","context":"Automatic identification and short data reports from vessels and shore stations.","wiki":"https://www.sigidwiki.com/wiki/Automatic_Identification_System_(AIS)","frequency_note":"Marine VHF channels  AIS 1 and AIS 2."},
    {"id":"dmr","name":"DMR","family":"FSK","kind":"fsk","scope":"VHF/UHF · land mobile","modulation":"4-level FSK","look_for":"Regularly framed digital bursts with multiple discrete symbol levels; a radio may show two-slot activity.","pitfall":"A basic waterfall generally cannot identify the protocol by itself; timing and decoder evidence help.","context":"Digital mobile radio voice and data.","wiki":"https://www.sigidwiki.com/wiki/DMR","frequency_note":"Used across licensed VHF and UHF land-mobile allocations."},
    {"id":"pocsag","name":"POCSAG pager","family":"FSK","kind":"fsk","scope":"VHF/UHF · paging","modulation":"2-FSK","look_for":"Longer one-way data bursts with a stable pair of tones and repeated message framing.","pitfall":"Other paging formats may look similar; exact identification benefits from bit timing and a protocol decoder.","context":"Legacy and continuing paging networks.","wiki":"https://www.sigidwiki.com/wiki/POCSAG","frequency_note":"Regional paging allocations vary.","lookup":{"bandwidth_hz":[500,3000],"tone_count":[2,2]}},
    {"id":"noaa-apt","name":"NOAA APT weather image","family":"Image","kind":"image","scope":"137 MHz · satellite","modulation":"FM subcarrier carrying a line image","look_for":"A long continuous pass with repeated image lines and broad, slowly changing horizontal structure.","pitfall":"A partial pass or incorrect audio level can make the image look washed out or compressed.","context":"Analog weather imagery from polar-orbiting satellites.","wiki":"https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)","frequency_note":"NOAA APT downlinks are in the 137 MHz range."},
    {"id":"wefax","name":"Weather fax (WEFAX)","family":"Image","kind":"image","scope":"HF · maritime/weather","modulation":"Single-sideband audio with frequency-shifted image tone","look_for":"A long, regular sweep with repeated narrow synchronization bars and gradual image changes.","pitfall":"Tuning error, drift, or a wrong line rate can skew the image; inspect the stable synchronization marks.","context":"Weather charts and maritime information transmitted as facsimile.","wiki":"https://www.sigidwiki.com/wiki/Radiofax","frequency_note":"HF frequencies vary by station and schedule."},
    {"id":"rtty","name":"RTTY","family":"FSK","kind":"fsk","scope":"HF · amateur / utility","modulation":"2-FSK","look_for":"Two steady mark/space tones that switch in a regular pattern; often a narrow pair of parallel traces.","pitfall":"Two simultaneous stations can resemble the pair. Switching transitions are the stronger clue.","context":"Baudot text links, amateur contacts, and legacy utility traffic.","wiki":"https://sigidwiki.com/wiki/Radio_Teletype_(RTTY)","frequency_note":"Found across HF; use current band plans and regional allocations.","lookup":{"bandwidth_hz":[100,350],"tone_count":[2,2]}},
    {"id":"cw","name":"Morse (CW)","family":"Morse","kind":"morse","scope":"HF/VHF · amateur / utility","modulation":"On-off keyed carrier","look_for":"A narrow carrier switching into dots and dashes, with gaps at multiple characteristic lengths.","pitfall":"A constant carrier with fading or interference can create false breaks; check the rhythm over a longer interval.","context":"Amateur and utility radiotelegraphy.","wiki":"https://www.sigidwiki.com/wiki/Continuous_Wave_(CW)","frequency_note":"Used across many amateur and utility bands."},
    {"id":"dsc","name":"Digital Selective Calling (DSC)","family":"MFSK","kind":"mfsk","scope":"HF/VHF · maritime","modulation":"Two-tone FSK","look_for":"A structured call burst with alternating tone energy and a recognizable fixed-duration framing pattern.","pitfall":"Do not decode or retransmit distress traffic casually; use a proper receiver workflow and applicable procedures.","context":"Maritime calling and safety signaling.","wiki":"https://www.sigidwiki.com/wiki/DSC","frequency_note":"Uses designated maritime HF and VHF channels.","lookup":{"bandwidth_hz":[100,400],"tone_count":[2,2]}},
]


def build_field_guide() -> dict[str, Any]:
    catalog = json.loads(files(__package__).joinpath("mode_catalog.json").read_text(encoding="utf-8"))
    raw_by_id = {item["id"]: item for item in catalog}
    fldigi = []
    for mode in catalog:
        if not str(mode.get("id", "")).startswith("fldigi-"):
            continue
        family = mode.get("family", "Digital")
        guidance = FAMILY_GUIDANCE.get(family, FAMILY_GUIDANCE["Digital"])
        raw = raw_by_id.get(mode["id"], {})
        entry = {
            "id": mode["id"], "name": mode["display_name"], "family": family,
            "kind": guidance["render"], "scope": "Fldigi mode · commonly used on HF",
            "modulation": mode.get("method") or family,
            "look_for": guidance["look_for"], "pitfall": guidance["pitfall"],
            "context": f"Fldigi selectable mode: {mode.get('protocol') or mode.get('variant') or mode['display_name']}.",
            "frequency_note": raw.get("source_frequencies") or "Operating frequency depends on the band, activity, and station convention.",
            "wiki": raw.get("source_url"), "variant": mode.get("variant"),
            "decoder_status": mode.get("decoder_status", "Fldigi selectable; decode validation varies by mode."),
            "catalog_source": mode.get("catalog_source"), "aliases": mode.get("aliases", []),
        }
        match = re.fullmatch(r"fldigi-olivia-(\d+)-(\d+)", mode["id"])
        if match:
            tone_count, bandwidth = (int(value) for value in match.groups())
            entry["lookup"] = {
                "bandwidth_hz": [bandwidth * 0.8, bandwidth * 1.2],
                "tone_count": [tone_count, tone_count],
            }
        fldigi.append(entry)
    return {"version": 1, "entries": [*COMMON_SIGNALS, *fldigi], "families": sorted({item["family"] for item in [*COMMON_SIGNALS, *fldigi]})}
