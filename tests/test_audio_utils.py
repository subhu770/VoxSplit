"""
Unit Tests for VoxSplit Audio Processing Utilities.
"""

import io
import numpy as np
import pytest
import soundfile as sf

from app.core.audio_utils import (
    load_audio_from_bytes,
    resample_audio,
    float_to_int16,
    int16_to_float,
    pcm_bytes_to_float_array,
    float_array_to_pcm_bytes,
    compute_rms_energy_dbfs,
    compute_zero_crossing_rate,
    normalize_audio,
    slice_audio,
)


def test_float_to_int16_and_back():
    """Verifies precision and bounds handling in float32 <-> int16 PCM conversions."""
    original = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    int16_arr = float_to_int16(original)

    assert int16_arr.dtype == np.int16
    assert int16_arr[0] == -32767
    assert int16_arr[-1] == 32767

    recovered = int16_to_float(int16_arr)
    assert recovered.dtype == np.float32
    np.testing.assert_allclose(recovered[1:-1], [-1.0, -0.5, 0.0, 0.5, 1.0], atol=0.001)


def test_pcm_bytes_roundtrip():
    """Verifies byte serialization between raw PCM and float32 arrays."""
    audio = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
    pcm_bytes = float_array_to_pcm_bytes(audio)
    assert len(pcm_bytes) == 2000

    recovered = pcm_bytes_to_float_array(pcm_bytes)
    assert len(recovered) == 1000
    np.testing.assert_allclose(audio, recovered, atol=0.001)


def test_load_audio_from_bytes_valid_wav():
    """Verifies that soundfile decodes generated WAV bytes into mono float32 at 16kHz."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    wav_bytes = buf.getvalue()

    decoded, target_sr = load_audio_from_bytes(wav_bytes, target_sr=16000)
    assert target_sr == 16000
    assert len(decoded) == 16000
    assert decoded.dtype == np.float32
    assert np.max(np.abs(decoded)) <= 1.0


def test_load_audio_from_bytes_empty():
    """Ensures empty byte input raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        load_audio_from_bytes(b"")


def test_load_audio_from_bytes_corrupted():
    """Ensures corrupt byte buffer raises ValueError."""
    with pytest.raises(ValueError, match="Could not parse"):
        load_audio_from_bytes(b"NotAnAudioFileHeader")


def test_resample_audio():
    """Verifies rate conversion preserves signal duration."""
    orig_sr = 32000
    target_sr = 16000
    audio = np.ones(orig_sr, dtype=np.float32)  # 1.0 second

    resampled = resample_audio(audio, orig_sr, target_sr)
    assert len(resampled) == target_sr


def test_compute_rms_energy_dbfs():
    """Checks dBFS computation for silence vs normalized tone."""
    silence = np.zeros(1600, dtype=np.float32)
    assert compute_rms_energy_dbfs(silence) <= -90.0

    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = np.sin(2 * np.pi * 400 * t).astype(np.float32)
    # RMS of unit sine wave is 1/sqrt(2) = ~0.707 -> 20*log10(0.707) = -3.01 dBFS
    db = compute_rms_energy_dbfs(sine)
    assert pytest.approx(db, abs=0.2) == -3.01


def test_compute_zero_crossing_rate():
    """Higher frequency signal must produce higher zero-crossing rate."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    high_freq = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    zcr_low = np.mean(compute_zero_crossing_rate(low_freq))
    zcr_high = np.mean(compute_zero_crossing_rate(high_freq))
    assert zcr_high > zcr_low


def test_normalize_audio():
    """Verifies RMS gain normalizer scales low amplitude audio towards target dB."""
    audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.05
    orig_db = compute_rms_energy_dbfs(audio)

    normalized = normalize_audio(audio, target_db=-18.0)
    new_db = compute_rms_energy_dbfs(normalized)

    assert new_db > orig_db
    assert np.max(np.abs(normalized)) <= 1.0


def test_slice_audio():
    """Verifies slice_audio bounds clamping and edge cases."""
    audio = np.arange(16000, dtype=np.float32)  # 1.0s at 16kHz
    sub = slice_audio(audio, start_sec=0.25, end_sec=0.75, sample_rate=16000)

    assert len(sub) == 8000
    assert sub[0] == 4000
    assert sub[-1] == 11999

    # Out-of-bounds start returns empty
    empty_sub = slice_audio(audio, start_sec=2.0, end_sec=3.0, sample_rate=16000)
    assert len(empty_sub) == 0

