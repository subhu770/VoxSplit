"""
Unit Tests for VoxSplit Voice Activity Detection (VAD) Service.
"""

import numpy as np
import pytest

from app.core.config import Settings
from app.services.vad import VoiceActivityDetector, SpeechInterval
from samples.generate_synthetic_sample import generate_harmonic_voice


def test_vad_rejects_pure_silence(silent_waveform, sample_rate):
    """Verifies that purely silent waveforms produce zero speech intervals."""
    detector = VoiceActivityDetector()
    segments = detector.detect_speech_segments(silent_waveform, sample_rate=sample_rate)
    assert segments == []


def test_vad_rejects_low_amplitude_noise(noise_waveform, sample_rate):
    """Verifies that background noise below the energy threshold is safely ignored."""
    detector = VoiceActivityDetector()
    segments = detector.detect_speech_segments(noise_waveform, sample_rate=sample_rate)
    assert segments == []


def test_vad_detects_voiced_intervals(sample_rate):
    """Verifies that clear voiced speech surrounded by silence is accurately bounded."""
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    speech = generate_harmonic_voice(
        duration_sec=1.5,
        fundamental_hz=150.0,
        formant_peaks=[600.0, 1100.0, 2500.0],
        sample_rate=sample_rate,
        amplitude=0.6
    )

    audio = np.concatenate([silence, speech, silence])
    detector = VoiceActivityDetector()
    segments = detector.detect_speech_segments(audio, sample_rate=sample_rate)

    assert len(segments) >= 1
    # Speech starts at 0.5s and ends around 2.0s
    start, end = segments[0]
    assert 0.3 <= start <= 0.8
    assert 1.8 <= end <= 2.3
    assert (end - start) >= 1.0


def test_vad_filters_micro_clicks(sample_rate):
    """Ensures short impulse clicks shorter than min_speech_duration_ms are discarded."""
    audio = np.zeros(int(1.0 * sample_rate), dtype=np.float32)
    # Inject a 30ms pop
    click_len = int(0.030 * sample_rate)
    audio[2000:2000 + click_len] = 0.8

    detector = VoiceActivityDetector()
    segments = detector.detect_speech_segments(audio, sample_rate=sample_rate)
    assert segments == []


def test_speech_interval_properties():
    """Validates SpeechInterval data class attributes."""
    interval = SpeechInterval(start=1.2351, end=4.5678)
    assert interval.duration == pytest.approx(3.3327, abs=0.001)
    assert interval.to_tuple() == (1.235, 4.568)
    assert "SpeechInterval" in repr(interval)


def test_vad_splits_long_segments(sample_rate):
    """Verifies that continuous segments longer than max_speech_duration are partitioned."""
    settings = Settings(VAD_MAX_SPEECH_DURATION_SEC=3.0)
    detector = VoiceActivityDetector(settings=settings)

    # 7.0 seconds continuous speech
    long_speech = generate_harmonic_voice(
        duration_sec=7.0,
        fundamental_hz=160.0,
        formant_peaks=[700.0, 1200.0],
        sample_rate=sample_rate,
        amplitude=0.6
    )
    segments = detector.detect_speech_segments(long_speech, sample_rate=sample_rate)

    assert len(segments) >= 2
    for start, end in segments:
        duration = end - start
        assert duration <= 3.5
