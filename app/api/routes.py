"""
VoxSplit REST API Endpoints.
Production endpoints for audio transcription, speaker diarization, and health diagnostics.
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
import numpy as np

from app.core.config import get_settings
from app.core.audio_utils import load_audio_from_bytes, slice_audio
from app.models.schemas import (
    TranscriptionResponse,
    DiarizedUtterance,
    DiarizationResponse,
    HealthResponse,
    WordTiming,
)
from app.services.vad import VoiceActivityDetector
from app.services.diarizer import SpeakerDiarizer
from app.services.asr import ASRService

logger = logging.getLogger("voxsplit.routes")

router = APIRouter()

# Instantiate service singletons / components
vad_detector = VoiceActivityDetector()
diarizer = SpeakerDiarizer()
asr_service = ASRService()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Returns system status, active inference runtime parameters, and Whisper model residency.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        model_loaded=asr_service.is_loaded,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        whisper_model_tier=settings.WHISPER_MODEL_SIZE,
    )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Audio Processing"],
    summary="Transcribe and Diarize Audio File"
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Uploaded audio file (WAV, FLAC, MP3, etc.)"),
    language: Optional[str] = Query(default=None, description="Enforce 2-letter ISO language code (e.g. 'en')")
) -> TranscriptionResponse:
    """
    End-to-End Pipeline:
    1. Ingest and normalize audio buffer to 16kHz mono float32.
    2. Run WebRTC/Energy Voice Activity Detection to extract voiced speech intervals.
    3. Run Acoustic Diarization to assign consistent speaker labels (SPEAKER_00, SPEAKER_01).
    4. Execute faster-whisper int8 speech-to-text with globally aligned word timestamps.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    # 1. Read & Validate Upload
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {str(e)}"
        )

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if len(audio_bytes) > settings.AUDIO_MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum permissible size of {settings.AUDIO_MAX_FILE_SIZE_BYTES} bytes."
        )

    # 2. Decode and Resample to 16kHz float32
    try:
        audio, sr = load_audio_from_bytes(audio_bytes, target_sr=settings.SAMPLE_RATE)
    except Exception as e:
        logger.error("Audio decoding error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio decoding failure: {str(e)}"
        )

    total_duration = round(len(audio) / sr, 3)

    # 3. Voice Activity Detection
    try:
        speech_intervals = vad_detector.detect_speech_segments(audio, sample_rate=sr)
    except Exception as e:
        logger.error("VAD execution error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VAD segmentation failure: {str(e)}"
        )

    # Handle pure silence / no speech detected
    if not speech_intervals:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return TranscriptionResponse(
            utterances=[],
            full_text="",
            total_duration=total_duration,
            speakers_detected=[],
            processing_time_ms=round(elapsed_ms, 2),
            language=language
        )

    # 4. Speaker Diarization
    try:
        diarized_segments = diarizer.diarize_segments(audio, speech_intervals)
    except Exception as e:
        logger.error("Diarization error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diarization failure: {str(e)}"
        )

    # 5. ASR Transcription for each diarized segment
    utterances: list[DiarizedUtterance] = []
    detected_languages = []

    for seg in diarized_segments:
        chunk = slice_audio(audio, seg.start, seg.end, sample_rate=sr)
        if len(chunk) == 0:
            continue

        try:
            text, conf, words, seg_lang = asr_service.transcribe_chunk(chunk, language=language)
            if seg_lang:
                detected_languages.append(seg_lang)
        except Exception as e:
            logger.error("ASR transcription failed on segment [%.2f - %.2f]: %s", seg.start, seg.end, e)
            text, conf, words = "", 0.0, []

        # Re-align word timestamps relative to overall stream timeline
        global_words: list[WordTiming] = [
            WordTiming(
                word=w.word,
                start=round(seg.start + w.start, 3),
                end=round(seg.start + w.end, 3),
                probability=w.probability
            )
            for w in words
        ]

        if text:  # Only append non-empty utterances
            utterances.append(
                DiarizedUtterance(
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    confidence=conf,
                    words=global_words
                )
            )

    # Compile aggregated metadata
    full_text = " ".join(u.text for u in utterances).strip()
    speakers_detected = sorted(list({u.speaker for u in utterances}))
    primary_language = detected_languages[0] if detected_languages else language
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return TranscriptionResponse(
        utterances=utterances,
        full_text=full_text,
        total_duration=total_duration,
        speakers_detected=speakers_detected,
        processing_time_ms=round(elapsed_ms, 2),
        language=primary_language
    )


@router.post(
    "/diarize",
    response_model=DiarizationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Audio Processing"],
    summary="Pure Acoustic Speaker Diarization"
)
async def diarize_audio(
    file: UploadFile = File(..., description="Uploaded audio file (WAV, FLAC, MP3, etc.)")
) -> DiarizationResponse:
    """
    Rapid acoustic-only speaker segmentation without speech-to-text inference.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        audio, sr = load_audio_from_bytes(audio_bytes, target_sr=settings.SAMPLE_RATE)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Audio error: {e}")

    total_duration = round(len(audio) / sr, 3)
    speech_intervals = vad_detector.detect_speech_segments(audio, sample_rate=sr)
    diarized_segments = diarizer.diarize_segments(audio, speech_intervals)

    speakers = sorted(list({seg.speaker for seg in diarized_segments}))
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return DiarizationResponse(
        segments=diarized_segments,
        speakers=speakers,
        total_duration=total_duration,
        processing_time_ms=round(elapsed_ms, 2)
    )

