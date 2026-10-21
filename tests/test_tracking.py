from __future__ import annotations

from sigid.tracking import LiveTrackRegistry


def candidate(center_hz: float, *, snr: float = 20.0) -> dict:
    return {
        "center_hz": center_hz,
        "bandwidth_hz": 400.0,
        "peak_snr_db": snr,
        "protocol": None,
    }


def test_live_registry_keeps_ids_stable_and_confirms_only_new_observations():
    registry = LiveTrackRegistry("session-a", confirmation_observations=3)
    first = registry.update([candidate(1500)], sample_position=1000, sample_rate=1000)
    repeated_window = registry.update([candidate(1500)], sample_position=1000, sample_rate=1000)
    drifted = registry.update([candidate(1540)], sample_position=2000, sample_rate=1000)
    confirmed = registry.update([candidate(1560)], sample_position=3000, sample_rate=1000)

    assert first[0]["track_id"] == repeated_window[0]["track_id"] == drifted[0]["track_id"] == confirmed[0]["track_id"]
    assert repeated_window[0]["track_observations"] == 1
    assert drifted[0]["track_state"] == "provisional"
    assert confirmed[0]["track_state"] == "persistent"
    assert confirmed[0]["track_age_seconds"] == 2.0
    assert confirmed[0]["protocol"] is None


def test_live_registry_separates_tracks_expires_stale_and_caps_count():
    registry = LiveTrackRegistry("session-b", max_tracks=2, expiry_seconds=2)
    initial = registry.update(
        [candidate(1000), candidate(4000)], sample_position=0, sample_rate=1000
    )
    assert len({item["track_id"] for item in initial}) == 2

    moved = registry.update(
        [candidate(1010), candidate(4010), candidate(7000, snr=30)],
        sample_position=1000,
        sample_rate=1000,
    )
    assert len(moved) == 2
    assert moved[0]["center_hz"] == 1010
    assert moved[1]["center_hz"] == 7000

    after_expiry = registry.update(
        [candidate(1010)], sample_position=5000, sample_rate=1000
    )
    assert after_expiry[0]["track_id"] != moved[0]["track_id"]
