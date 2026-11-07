"""
Unit Tests for VoxSplit Pydantic v2 Schemas and Contracts.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    WordTiming,
    Segment,
    DiarizedUtterance,
    TranscriptionResponse,
    DiarizationSegment,
    DiarizationResponse,
    HealthResponse,
    StreamingTranscriptChunk,
)


def test_segment_schema_computed_duration():
    """Checks Segment model computed duration property."""
    seg = Segment(start=1.5, end=4.25, text="Testing audio segmentation")
    assert seg.duration == 2.75
    assert seg.confidence == 1.0


def test_word_timing_validation():
    """Checks WordTiming constraints and serialization."""
    word = WordTiming(word="hello", start=0.5, end=0.8, probability=0.98)
    assert word.word == "hello"
    assert word.probability == 0.98

    # Negative timestamps should raise ValidationError
    with pytest.raises(ValidationError):
        WordTiming(word="fail", start=-1.0, end=0.5)


def test_diarized_utterance_schema():
    """Checks DiarizedUtterance schema with nested word alignments."""
    utterance = DiarizedUtterance(
        speaker="SPEAKER_00",
        start=0.0,
        end=2.5,
        text="Hello world speech recognition",
        confidence=0.92,
        words=[
            WordTiming(word="Hello", start=0.0, end=0.5, probability=0.95),
            WordTiming(word="world", start=0.5, end=1.0, probability=0.90),
        ]
    )
    assert utterance.speaker == "SPEAKER_00"
    assert utterance.duration == 2.5
    assert len(utterance.words) == 2

    # JSON roundtrip
    dumped = utterance.model_dump_json()
    assert "SPEAKER_00" in dumped
    reconstructed = DiarizedUtterance.model_validate_json(dumped)
    assert reconstructed == utterance


def test_transcription_response_schema():
    """Checks TranscriptionResponse aggregation schema."""
    resp = TranscriptionResponse(
        utterances=[
            DiarizedUtterance(
                speaker="SPEAKER_00",
                start=0.0,
                end=1.5,
                text="Turn one",
                confidence=0.9
            ),
            DiarizedUtterance(
                speaker="SPEAKER_01",
                start=2.0,
                end=3.5,
                text="Turn two",
                confidence=0.88
            ),
        ],
        full_text="Turn one Turn two",
        total_duration=3.5,
        speakers_detected=["SPEAKER_00", "SPEAKER_01"],
        processing_time_ms=124.5,
        language="en"
    )
    assert resp.total_duration == 3.5
    assert len(resp.speakers_detected) == 2
    assert resp.utterances[0].speaker == "SPEAKER_00"
    assert resp.utterances[1].speaker == "SPEAKER_01"


def test_health_response_schema():
    """Checks HealthResponse serialization."""
    health = HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=True,
        device="cpu",
        compute_type="int8",
        whisper_model_tier="tiny"
    )
    data = health.model_dump()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_streaming_chunk_schema():
    """Checks StreamingTranscriptChunk structure."""
    chunk = StreamingTranscriptChunk(
        event="final",
        speaker="SPEAKER_00",
        start=0.0,
        end=2.1,
        text="Recognized voice chunk",
        is_final=True
    )
    assert chunk.is_final is True
    assert chunk.event == "final"

