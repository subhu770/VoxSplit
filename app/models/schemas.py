"""
VoxSplit Pydantic v2 Data Transfer Models and Response Schemas.
Strict schema definitions for audio processing, diarization, and ASR pipelines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, computed_field


class WordTiming(BaseModel):
    """Word-level alignment metadata from ASR decoder."""
    word: str = Field(..., description="Recognized token or word")
    start: float = Field(..., ge=0.0, description="Start timestamp relative to stream origin in seconds")
    end: float = Field(..., ge=0.0, description="End timestamp in seconds")
    probability: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class Segment(BaseModel):
    """Basic acoustic or transcription slice."""
    start: float = Field(..., ge=0.0, description="Segment start timestamp in seconds")
    end: float = Field(..., ge=0.0, description="Segment end timestamp in seconds")
    text: Optional[str] = Field(default="", description="Transcribed textual content")
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="ASR decoding confidence score")

    @computed_field
    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return round(max(0.0, self.end - self.start), 3)


class DiarizedUtterance(BaseModel):
    """
    Time-aligned transcribed utterance tagged with an assigned speaker identity.
    """
    speaker: str = Field(..., description="Unique speaker cluster identifier (e.g. SPEAKER_00)")
    start: float = Field(..., ge=0.0, description="Utterance start timestamp in seconds")
    end: float = Field(..., ge=0.0, description="Utterance end timestamp in seconds")
    text: str = Field(..., description="Recognized speech text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Aggregated confidence score")
    words: List[WordTiming] = Field(default_factory=list, description="Word-level time alignment tokens")

    @computed_field
    @property
    def duration(self) -> float:
        """Utterance duration in seconds."""
        return round(max(0.0, self.end - self.start), 3)


class TranscriptionResponse(BaseModel):
    """
    Primary API response schema for end-to-end VAD + ASR + Diarization pipeline.
    """
    utterances: List[DiarizedUtterance] = Field(
        ...,
        description="Chronologically sorted list of diarized spoken utterances"
    )
    full_text: str = Field(..., description="Complete aggregated transcript")
    total_duration: float = Field(..., ge=0.0, description="Total audio duration in seconds")
    speakers_detected: List[str] = Field(..., description="List of unique speaker IDs discovered")
    processing_time_ms: float = Field(..., ge=0.0, description="Total server processing latency in milliseconds")
    language: Optional[str] = Field(default=None, description="Detected or requested transcription language")


class DiarizationSegment(BaseModel):
    """A pure acoustic diarization turn without transcript."""
    speaker: str = Field(..., description="Assigned speaker identity (e.g. SPEAKER_00)")
    start: float = Field(..., ge=0.0, description="Voiced segment start time in seconds")
    end: float = Field(..., ge=0.0, description="Voiced segment end time in seconds")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Speaker cluster similarity score")

    @computed_field
    @property
    def duration(self) -> float:
        """Segment length in seconds."""
        return round(max(0.0, self.end - self.start), 3)


class DiarizationResponse(BaseModel):
    """Response payload for pure speaker diarization endpoint."""
    segments: List[DiarizationSegment] = Field(..., description="Speaker activity turns")
    speakers: List[str] = Field(..., description="Unique speaker IDs present in the audio")
    total_duration: float = Field(..., ge=0.0, description="Audio file duration in seconds")
    processing_time_ms: float = Field(..., ge=0.0, description="Computation time in ms")


class HealthResponse(BaseModel):
    """Service health inspection schema."""
    status: str = Field(default="ok", description="Operational status")
    version: str = Field(..., description="VoxSplit version")
    model_loaded: bool = Field(..., description="Whether faster-whisper model is currently resident in RAM")
    device: str = Field(..., description="Active compute device ('cpu' or 'cuda')")
    compute_type: str = Field(..., description="Active quantization format ('int8', 'float16')")
    whisper_model_tier: str = Field(..., description="Configured Whisper model tier")


class StreamingTranscriptChunk(BaseModel):
    """WebSocket streaming event payload."""
    event: str = Field(..., description="Event type: 'speech_start', 'partial', 'final', 'speech_end', 'error'")
    speaker: Optional[str] = Field(default=None, description="Speaker tag for final utterances")
    start: float = Field(default=0.0, ge=0.0, description="Relative timestamp start in seconds")
    end: float = Field(default=0.0, ge=0.0, description="Relative timestamp end in seconds")
    text: str = Field(default="", description="Recognized text fragment")
    is_final: bool = Field(default=False, description="Whether this utterance is finalized")

