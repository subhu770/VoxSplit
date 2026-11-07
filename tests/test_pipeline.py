"""
End-to-End Integration Tests for VoxSplit REST & WebSocket Pipelines.
"""

import io
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import soundfile as sf
import numpy as np

from app.models.schemas import WordTiming


def test_health_endpoints(client: TestClient):
    """Verifies both root /health and /api/v1/health return 200 OK."""
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "device" in data
        assert "compute_type" in data


def test_root_endpoint(client: TestClient):
    """Checks root endpoint metadata."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "VoxSplit"


def test_transcribe_empty_file_returns_400(client: TestClient):
    """Verifies that uploading an empty audio file returns HTTP 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_transcribe_corrupted_file_returns_422(client: TestClient):
    """Verifies that uploading an invalid audio byte stream returns HTTP 422."""
    files = {"file": ("bad.wav", b"INVALID_RIFF_AUDIO_DATA_BYTES", "audio/wav")}
    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 422
    assert "Audio decoding failure" in response.json()["detail"]


def test_transcribe_pure_silence(client: TestClient, silent_waveform, sample_rate):
    """Verifies that purely silent audio returns empty utterances and 0 detected speakers."""
    buf = io.BytesIO()
    sf.write(buf, silent_waveform, sample_rate, format="WAV")
    files = {"file": ("silence.wav", buf.getvalue(), "audio/wav")}

    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["utterances"] == []
    assert data["full_text"] == ""
    assert data["speakers_detected"] == []
    assert data["total_duration"] == pytest.approx(2.0, abs=0.1)


def test_diarize_endpoint_success(client: TestClient, multi_speaker_wav_bytes):
    """Verifies that POST /api/v1/diarize correctly segments and tags multiple speakers."""
    files = {"file": ("dialogue.wav", multi_speaker_wav_bytes, "audio/wav")}
    response = client.post("/api/v1/diarize", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert "speakers" in data
    assert len(data["segments"]) >= 2
    assert len(data["speakers"]) >= 2
    assert data["total_duration"] > 5.0
    assert data["processing_time_ms"] > 0


def test_end_to_end_transcribe_pipeline(client: TestClient, multi_speaker_wav_bytes):
    """
    Verifies full end-to-end VAD + Diarization + ASR pipeline with mock/deterministic ASR output
    to validate time-stamping, speaker alignment, and JSON response contracts.
    """
    mock_responses = [
        ("Hello, this is the first speaker speaking.", 0.94, [
            WordTiming(word="Hello,", start=0.0, end=0.4, probability=0.96),
            WordTiming(word="speaker", start=0.5, end=0.9, probability=0.92)
        ], "en"),
        ("And I am the second speaker joining.", 0.91, [
            WordTiming(word="And", start=0.0, end=0.3, probability=0.93),
            WordTiming(word="second", start=0.4, end=0.8, probability=0.90)
        ], "en"),
        ("Continuing the conversation here.", 0.95, [], "en")
    ]

    with patch("app.services.asr.ASRService.transcribe_chunk", side_effect=mock_responses):
        files = {"file": ("dialogue.wav", multi_speaker_wav_bytes, "audio/wav")}
        response = client.post("/api/v1/transcribe?language=en", files=files)

        assert response.status_code == 200
        data = response.json()

        assert len(data["utterances"]) >= 2
        assert len(data["speakers_detected"]) >= 2
        assert "SPEAKER_00" in data["speakers_detected"]
        assert len(data["full_text"]) > 0

        # Verify global word timestamp offset alignment
        for utt in data["utterances"]:
            assert utt["start"] < utt["end"]
            for word in utt["words"]:
                # Word start timestamp must be within the utterance range
                assert word["start"] >= (utt["start"] - 0.05)
                assert word["end"] <= (utt["end"] + 0.1)


def test_websocket_streaming_flow(client: TestClient, sample_rate, synthetic_speaker_a):
    """
    Verifies WebSocket /api/v1/stream connection, JSON control actions (reset/flush),
    and binary PCM frame processing.
    """
    with client.websocket_connect("/api/v1/stream") as websocket:
        # 1. Send reset action
        websocket.send_text(json.dumps({"action": "reset"}))
        msg = websocket.receive_json()
        assert msg["event"] == "reset_ack"

        # 2. Stream 0.5s of audio as raw 16-bit PCM bytes
        pcm_bytes = (synthetic_speaker_a[:int(0.5 * sample_rate)] * 32767).astype(np.int16).tobytes()
        websocket.send_bytes(pcm_bytes)

        # 3. Request flush
        websocket.send_text(json.dumps({"action": "flush"}))
