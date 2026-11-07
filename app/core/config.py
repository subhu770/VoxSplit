"""
VoxSplit Configuration Management Module.
Centralized application settings loaded via Pydantic Settings with environment variable overrides.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application runtime configuration and audio pipeline hyperparameters.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Service Metadata
    APP_NAME: str = Field(default="VoxSplit", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application semver")
    DEBUG: bool = Field(default=False, description="Debug logging and profiling flag")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API route prefix")

    # Audio Pipeline Specifications
    SAMPLE_RATE: int = Field(default=16000, description="Target processing sampling rate (Hz)")
    CHANNELS: int = Field(default=1, description="Target mono channel count")
    AUDIO_MAX_FILE_SIZE_BYTES: int = Field(default=100 * 1024 * 1024, description="Max upload size (100MB)")

    # Voice Activity Detection (VAD) Hyperparameters
    VAD_FRAME_DURATION_MS: int = Field(default=30, description="Frame duration in ms for VAD analysis (10, 20, or 30)")
    VAD_AGGRESSIVENESS: int = Field(default=3, ge=0, le=3, description="WebRTC VAD aggressiveness mode (0-3)")
    VAD_PADDING_DURATION_MS: int = Field(default=300, description="Silence hangover padding window (ms)")
    VAD_MIN_SPEECH_DURATION_MS: int = Field(default=250, description="Minimum voiced segment length (ms)")
    VAD_MAX_SPEECH_DURATION_SEC: float = Field(default=28.0, description="Maximum single chunk duration in seconds before forced split")
    VAD_ENERGY_THRESHOLD_DB: float = Field(default=-42.0, description="Energy threshold floor in dBFS for hybrid VAD gating")

    # Speaker Diarization Hyperparameters
    SPEAKER_SIMILARITY_THRESHOLD: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cluster assignment to an existing speaker"
    )
    DIARIZATION_WINDOW_SEC: float = Field(default=1.5, description="Feature extraction window size in seconds")
    DIARIZATION_HOP_SEC: float = Field(default=0.75, description="Feature extraction sliding hop size in seconds")
    N_MFCC: int = Field(default=24, description="Number of Mel-frequency cepstral coefficients")
    MAX_SPEAKERS: int = Field(default=10, description="Max speaker profiles tracked in dynamic buffer")

    # Automatic Speech Recognition (faster-whisper / CTranslate2)
    WHISPER_MODEL_SIZE: str = Field(default="tiny", description="Whisper model tier (tiny, base, small, medium, large-v3)")
    WHISPER_DEVICE: str = Field(default="cpu", description="Compute device ('cpu' or 'cuda')")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", description="Quantized compute type ('int8', 'float16', 'float32')")
    WHISPER_BEAM_SIZE: int = Field(default=5, ge=1, description="Beam size for decoding")
    WHISPER_CPU_THREADS: int = Field(default=4, ge=1, description="CPU inference worker threads")
    WHISPER_LANGUAGE: Optional[str] = Field(default=None, description="Enforced language code (None = auto-detect)")
    WHISPER_WORD_TIMESTAMPS: bool = Field(default=True, description="Extract word-level timestamp metadata")

    # WebSocket Real-Time Streaming Configuration
    WS_BUFFER_CHUNK_SEC: float = Field(default=1.5, description="Minimum audio chunk duration before processing in stream")
    WS_SILENCE_TIMEOUT_SEC: float = Field(default=0.8, description="Silence window to trigger final utterance flush")


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached singleton application settings.
    """
    return Settings()

