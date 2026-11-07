"""
VoxSplit Pytest Configuration and Shared Test Fixtures.
Provides synthetic waveform generators, sample audio buffers, and FastAPI TestClients.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from samples.generate_synthetic_sample import generate_harmonic_voice


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provides test-specific configuration."""
    return Settings(
        APP_NAME="VoxSplit-Test",
        DEBUG=True,
        WHISPER_MODEL_SIZE="tiny",
        WHISPER_DEVICE="cpu",
        WHISPER_COMPUTE_TYPE="int8",
        VAD_AGGRESSIVENESS=2,
        VAD_PADDING_DURATION_MS=200,
        VAD_MIN_SPEECH_DURATION_MS=150,
        SPEAKER_SIMILARITY_THRESHOLD=0.65
    )


@pytest.fixture
def client() -> TestClient:
    """FastAPI synchronous test client."""
    return TestClient(app)


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def silent_waveform(sample_rate: int) -> np.ndarray:
    """Generates 2.0 seconds of absolute digital silence."""
    return np.zeros(int(2.0 * sample_rate), dtype=np.float32)


@pytest.fixture
def noise_waveform(sample_rate: int) -> np.ndarray:
    """Generates 1.5 seconds of low-amplitude white noise below VAD threshold."""
    rng = np.random.default_rng(seed=42)
    return (rng.normal(0, 0.001, int(1.5 * sample_rate))).astype(np.float32)


@pytest.fixture
def synthetic_speaker_a(sample_rate: int) -> np.ndarray:
    """Generates 2.0s of synthetic male-like voice (F0=130Hz)."""
    return generate_harmonic_voice(
        duration_sec=2.0,
        fundamental_hz=130.0,
        formant_peaks=[700.0, 1220.0, 2600.0],
        sample_rate=sample_rate,
        amplitude=0.6
    )


@pytest.fixture
def synthetic_speaker_b(sample_rate: int) -> np.ndarray:
    """Generates 2.0s of synthetic female-like voice (F0=240Hz)."""
    return generate_harmonic_voice(
        duration_sec=2.0,
        fundamental_hz=240.0,
        formant_peaks=[400.0, 1900.0, 2900.0],
        sample_rate=sample_rate,
        amplitude=0.6
    )


@pytest.fixture
def multi_speaker_wav_bytes(
    sample_rate: int,
    synthetic_speaker_a: np.ndarray,
    synthetic_speaker_b: np.ndarray
) -> bytes:
    """
    Creates an in-memory WAV file with 2 speakers interleaved with silence:
    - 0.3s silence
    - 2.0s Speaker A
    - 0.5s silence
    - 2.0s Speaker B
    - 0.5s silence
    - 2.0s Speaker A
    """
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    initial_silence = np.zeros(int(0.3 * sample_rate), dtype=np.float32)

    full_audio = np.concatenate([
        initial_silence,
        synthetic_speaker_a,
        silence,
        synthetic_speaker_b,
        silence,
        synthetic_speaker_a
    ])

    buf = io.BytesIO()
    sf.write(buf, full_audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

