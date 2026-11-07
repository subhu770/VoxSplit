"""
VoxSplit Audio Processing Utilities.
High-performance conversion, normalization, resampling, and waveform extraction routines.
"""

import io
import math
import logging
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("voxsplit.audio_utils")


def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Decodes an arbitrary audio byte buffer into a normalized mono float32 NumPy array at target_sr.

    Args:
        audio_bytes: In-memory raw audio file content (e.g., WAV, FLAC, OGG).
        target_sr: Target sampling rate in Hz (default: 16000 Hz).

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is 1D float32 normalized to [-1.0, 1.0].

    Raises:
        ValueError: If audio content is empty, corrupted, or unsupported.
    """
    if not audio_bytes:
        raise ValueError("Audio byte buffer is empty.")

    try:
        buffer = io.BytesIO(audio_bytes)
        audio_data, orig_sr = sf.read(buffer, dtype="float32", always_2d=False)
    except Exception as e:
        logger.error("Failed to decode audio buffer with soundfile: %s", str(e))
        raise ValueError(f"Could not parse audio file. Ensure valid audio format (e.g. WAV, FLAC): {e}") from e

    # Downmix multi-channel audio to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Flatten if needed and ensure contiguous float32
    audio_data = np.ascontiguousarray(audio_data.squeeze(), dtype=np.float32)

    if audio_data.size == 0:
        raise ValueError("Decoded audio contains zero samples.")

    # Resample if sample rate does not match target
    if orig_sr != target_sr:
        logger.debug("Resampling audio from %d Hz to %d Hz", orig_sr, target_sr)
        audio_data = resample_audio(audio_data, orig_sr=orig_sr, target_sr=target_sr)

    # Ensure range is strictly [-1.0, 1.0]
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val

    return audio_data, target_sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resamples a float32 1D audio array from orig_sr to target_sr using high-quality polyphase filtering.
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Converts a normalized float32 waveform in [-1.0, 1.0] to signed 16-bit PCM integer array.
    Clips out-of-range values to prevent integer wraparound distortion.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    """
    Converts signed 16-bit PCM integer waveform array to normalized float32 in [-1.0, 1.0].
    """
    return (audio.astype(np.float32) / 32768.0).astype(np.float32)


def pcm_bytes_to_float_array(pcm_bytes: bytes) -> np.ndarray:
    """
    Decodes raw 16-bit mono little-endian linear PCM byte stream into float32 array [-1.0, 1.0].
    """
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)
    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_to_float(int16_arr)


def float_array_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """
    Converts float32 audio waveform into raw 16-bit mono little-endian linear PCM byte string.
    """
    int16_arr = float_to_int16(audio)
    return int16_arr.tobytes()


def compute_rms_energy_dbfs(audio: np.ndarray, eps: float = 1e-9) -> float:
    """
    Computes global Root-Mean-Square (RMS) energy in decibels relative to full scale (dBFS).
    """
    if audio.size == 0:
        return -100.0
    mean_sq = float(np.mean(np.square(audio)))
    if mean_sq <= 1e-12:
        return -100.0
    rms = math.sqrt(mean_sq)
    return float(20.0 * math.log10(rms))


def compute_frame_energy_dbfs(
    audio: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """
    Computes short-time frame energy in dBFS across time frames.
    """
    if audio.size < frame_length:
        return np.array([compute_rms_energy_dbfs(audio)], dtype=np.float32)

    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-9)
    energy_db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return energy_db.astype(np.float32)


def compute_zero_crossing_rate(
    audio: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """
    Calculates zero-crossing rate (ZCR) per frame for unvoiced/voiced discrimination.
    """
    if audio.size < frame_length:
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0
        return np.array([zcr], dtype=np.float32)
    return librosa.feature.zero_crossing_rate(
        audio, frame_length=frame_length, hop_length=hop_length
    )[0].astype(np.float32)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Applies RMS gain normalization to bring speech energy towards target_db without clipping peaks.
    """
    current_db = compute_rms_energy_dbfs(audio)
    if current_db <= -90.0:
        return audio
    gain_db = target_db - current_db
    gain_linear = 10.0 ** (gain_db / 20.0)
    normalized = audio * gain_linear
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized.astype(np.float32)


def slice_audio(
    audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extracts a time-bounded subsegment from an audio array safely clamped to signal bounds.
    """
    total_samples = len(audio)
    start_idx = max(0, int(round(start_sec * sample_rate)))
    end_idx = min(total_samples, int(round(end_sec * sample_rate)))

    if start_idx >= end_idx or start_idx >= total_samples:
        return np.empty(0, dtype=np.float32)

    return audio[start_idx:end_idx]
