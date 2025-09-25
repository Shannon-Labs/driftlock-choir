"""Unit tests for vowel formant synthesis and analysis helpers."""

from __future__ import annotations

import numpy as np

from phy.formants import (
    FormantAnalysisResult,
    FormantSynthesisConfig,
    analyze_missing_fundamental,
    synthesize_formant_preamble,
)


def test_formant_waveform_is_normalized() -> None:
    sample_rate = 5_000_000.0
    length = 2048
    config = FormantSynthesisConfig(
        profile='A',
        fundamental_hz=25_000.0,
        harmonic_count=8,
        include_fundamental=False,
        formant_scale=1_000.0,
    )

    waveform, library = synthesize_formant_preamble(length, sample_rate, config)

    assert waveform.shape == (length,)
    norm = np.linalg.norm(waveform)
    assert np.isclose(norm, 1.0, atol=1e-6)
    assert config.profile.upper() in library


def test_missing_fundamental_recovers_profile() -> None:
    sample_rate = 5_000_000.0
    length = 4096
    config = FormantSynthesisConfig(
        profile='E',
        fundamental_hz=20_000.0,
        harmonic_count=10,
        include_fundamental=False,
        formant_scale=800.0,
    )

    waveform, library = synthesize_formant_preamble(length, sample_rate, config)
    segment = waveform[:1024]
    result = analyze_missing_fundamental(segment, sample_rate, list(library.values()))

    assert isinstance(result, FormantAnalysisResult)
    assert result.label == 'E'
    assert result.missing_fundamental_hz > 0.0
    assert result.dominant_hz > result.missing_fundamental_hz
