"""
VoxSplit Core Package: Configuration and Audio Processing Primitives.
"""

from app.core.config import Settings, get_settings
from app.core.audio_utils import (
    load_audio_from_bytes,
    resample_audio,
    float_to_int16,
    int16_to_float,
    pcm_bytes_to_float_array,
    float_array_to_pcm_bytes,
    compute_rms_energy_dbfs,
    compute_frame_energy_dbfs,
    compute_zero_crossing_rate,
    normalize_audio,
    slice_audio,
)
from app.core.preprocessor import AudioPreprocessor

__all__ = [
    "Settings",
    "get_settings",
    "AudioPreprocessor",
    "load_audio_from_bytes",
    "resample_audio",
    "float_to_int16",
    "int16_to_float",
    "pcm_bytes_to_float_array",
    "float_array_to_pcm_bytes",
    "compute_rms_energy_dbfs",
    "compute_frame_energy_dbfs",
    "compute_zero_crossing_rate",
    "normalize_audio",
    "slice_audio",
]
