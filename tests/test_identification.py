"""Regression checks for the audio evidence -> ranked trials link."""

from __future__ import annotations

import numpy as np
import pytest

from sigid.classifier import propose_loaded
from sigid.identification import decoder_shortlist, measure_audio_evidence


def test_audio_evidence_recovers_two_tones_without_claiming_a_protocol():
    rate = 12_000
    tones = np.where((np.arange(rate) // 1200) % 2 == 0, 900.0, 1300.0)
    phase = np.cumsum(2 * np.pi * tones / rate)
    audio = np.sin(phase)
    result = measure_audio_evidence(audio, rate, 700, 1500)

    assert result["available"] is True
    assert result["source"] == "bounded_audio_roi"
    assert result["selected_band_hz"] == [700.0, 1500.0]
    assert result["analyzed_samples"] <= 2 * rate
    assert result["tone_grid_regular"] is True
    assert len(result["candidate_tone_peaks_hz"]) >= 2
    assert any(abs(f - 900) < 25 for f in result["candidate_tone_peaks_hz"])
    assert any(abs(f - 1300) < 25 for f in result["candidate_tone_peaks_hz"])
    assert "no modulation or protocol is verified" in result["limitation"].lower()


def test_audio_evidence_rejects_invalid_roi_and_abstains_on_silence():
    with pytest.raises(ValueError, match="Nyquist"):
        measure_audio_evidence(np.ones(4096), 12000, 5000, 7000)
    with pytest.raises(ValueError, match="finite"):
        measure_audio_evidence([float("nan")] * 1024, 12000)
    assert measure_audio_evidence(np.zeros(4096), 12000)["available"] is False


def test_ranked_shortlist_preserves_nonwinning_family_and_validated_limit():
    proposal = {
        "family": "FSK",
        "ranked_candidates": [
            {"mode_id": "bpsk31", "family": "PSK", "relative_model_score": 0.46},
            {"mode_id": "rtty", "family": "FSK", "relative_model_score": 0.41},
            {"mode_id": "thor16", "family": "MFSK", "relative_model_score": 0.13},
        ],
    }
    assert decoder_shortlist(proposal) == ["BPSK31", "RTTY", "THOR16"]
    assert decoder_shortlist({"family": "FSK"}) == ["RTTY", "THOR16", "BPSK31"]
    with pytest.raises(ValueError, match="max_modes"):
        decoder_shortlist(proposal, max_modes=50)


def test_hierarchical_classifier_ranks_alternatives_when_it_abstains(monkeypatch):
    def fake_probs(_features, params):
        return np.array([{
            "family": [0.55, 0.45],
            "fsk": [1.0],
            "psk": [0.7, 0.3],
        }[params["name"]]], dtype=np.float64)

    monkeypatch.setattr("sigid.classifier.predict_probabilities", fake_probs)
    metadata = {
        "model_type": "hierarchical_family_experts",
        "families": ["FSK", "PSK"],
        "family_thresholds": {"FSK": 0.9, "PSK": 0.9},
        "mode_thresholds": {"FSK": 0.9, "PSK": 0.9},
        "mode_classes_by_family": {
            "FSK": ["rtty"], "PSK": ["bpsk31", "qpsk31"],
        },
        "mode_details": {},
    }
    params = {
        "family": {"name": "family"},
        "experts": {"FSK": {"name": "fsk"}, "PSK": {"name": "psk"}},
    }
    proposal = propose_loaded(np.zeros((1, 4)), metadata, params)

    assert proposal["abstained"] is True
    assert proposal["method_id"] is None
    assert [row["mode_id"] for row in proposal["ranked_candidates"]] == [
        "rtty", "bpsk31", "qpsk31",
    ]
    assert proposal["ranked_candidates"][0]["relative_model_score"] == pytest.approx(0.55)
    assert decoder_shortlist(proposal) == ["RTTY", "BPSK31", "THOR16"]
