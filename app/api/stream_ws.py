"""
VoxSplit Real-Time WebSocket Streaming Endpoint.
Low-latency streaming speech recognition and online speaker assignment over WebSockets.
"""

import json
import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.core.config import get_settings
from app.core.audio_utils import pcm_bytes_to_float_array, compute_rms_energy_dbfs
from app.models.schemas import StreamingTranscriptChunk
from app.services.vad import VoiceActivityDetector
from app.services.diarizer import SpeakerDiarizer
from app.services.asr import ASRService

logger = logging.getLogger("voxsplit.stream_ws")

router = APIRouter()


@router.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """
    Bi-directional streaming transcription & diarization over WebSockets.

    Accepts:
        - Binary frames: Raw 16-bit signed PCM audio at 16,000 Hz mono.
        - Text frames: JSON control messages, e.g. {"action": "flush"} or {"action": "reset"}.

    Yields:
        - JSON messages conforming to StreamingTranscriptChunk schema.
    """
    await websocket.accept()
    settings = get_settings()

    # Per-session local state
    vad = VoiceActivityDetector(settings)
    diarizer = SpeakerDiarizer(settings)
    asr = ASRService(settings)

    sample_rate = settings.SAMPLE_RATE
    min_chunk_samples = int(settings.WS_BUFFER_CHUNK_SEC * sample_rate)
    silence_timeout_samples = int(settings.WS_SILENCE_TIMEOUT_SEC * sample_rate)

    audio_buffer = np.empty(0, dtype=np.float32)
    stream_position_samples = 0
    in_speech = False
    speech_start_sec = 0.0
    silence_counter_samples = 0
    current_utterance_audio = np.empty(0, dtype=np.float32)

    logger.info("WebSocket streaming client connected.")

    try:
        while True:
            message = await websocket.receive()

            # Handle control commands via JSON text frames
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    if action == "reset":
                        audio_buffer = np.empty(0, dtype=np.float32)
                        current_utterance_audio = np.empty(0, dtype=np.float32)
                        stream_position_samples = 0
                        in_speech = False
                        diarizer.reset()
                        await websocket.send_text(
                            StreamingTranscriptChunk(
                                event="reset_ack",
                                text="Session reset successful."
                            ).model_dump_json()
                        )
                        continue
                    elif action == "flush" and len(current_utterance_audio) > 0:
                        # Force finalize any pending buffered speech
                        end_sec = stream_position_samples / sample_rate
                        spk, conf = diarizer.assign_speaker_online(
                            current_utterance_audio,
                            len(current_utterance_audio) / sample_rate
                        )
                        text, _, _, _ = asr.transcribe_chunk(current_utterance_audio)
                        if text:
                            await websocket.send_text(
                                StreamingTranscriptChunk(
                                    event="final",
                                    speaker=spk,
                                    start=round(speech_start_sec, 3),
                                    end=round(end_sec, 3),
                                    text=text,
                                    is_final=True
                                ).model_dump_json()
                            )
                        current_utterance_audio = np.empty(0, dtype=np.float32)
                        in_speech = False
                        continue
                except json.JSONDecodeError:
                    pass

            # Handle binary PCM audio frames
            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                chunk_float = pcm_bytes_to_float_array(chunk_bytes)
                if len(chunk_float) == 0:
                    continue

                audio_buffer = np.concatenate([audio_buffer, chunk_float])
                chunk_len = len(chunk_float)
                stream_position_samples += chunk_len

                # Quick energy check for voice activity
                chunk_db = compute_rms_energy_dbfs(chunk_float)
                is_voiced = chunk_db > settings.VAD_ENERGY_THRESHOLD_DB

                if is_voiced:
                    if not in_speech:
                        in_speech = True
                        speech_start_sec = (stream_position_samples - chunk_len) / sample_rate
                        await websocket.send_text(
                            StreamingTranscriptChunk(
                                event="speech_start",
                                start=round(speech_start_sec, 3),
                                text=""
                            ).model_dump_json()
                        )

                    silence_counter_samples = 0
                    current_utterance_audio = np.concatenate([current_utterance_audio, chunk_float])

                    # Emit partial transcript when buffer grows large enough
                    if len(current_utterance_audio) >= min_chunk_samples:
                        partial_text, _, _, _ = asr.transcribe_chunk(current_utterance_audio)
                        if partial_text:
                            curr_end = stream_position_samples / sample_rate
                            await websocket.send_text(
                                StreamingTranscriptChunk(
                                    event="partial",
                                    start=round(speech_start_sec, 3),
                                    end=round(curr_end, 3),
                                    text=partial_text,
                                    is_final=False
                                ).model_dump_json()
                            )
                else:
                    if in_speech:
                        silence_counter_samples += chunk_len
                        current_utterance_audio = np.concatenate([current_utterance_audio, chunk_float])

                        # Silence timeout reached: finalize utterance
                        if silence_counter_samples >= silence_timeout_samples:
                            end_sec = stream_position_samples / sample_rate
                            duration = len(current_utterance_audio) / sample_rate

                            if duration >= (settings.VAD_MIN_SPEECH_DURATION_MS / 1000.0):
                                spk, conf = diarizer.assign_speaker_online(current_utterance_audio, duration)
                                final_text, _, _, _ = asr.transcribe_chunk(current_utterance_audio)

                                if final_text:
                                    await websocket.send_text(
                                        StreamingTranscriptChunk(
                                            event="final",
                                            speaker=spk,
                                            start=round(speech_start_sec, 3),
                                            end=round(end_sec, 3),
                                            text=final_text,
                                            is_final=True
                                        ).model_dump_json()
                                    )

                            in_speech = False
                            silence_counter_samples = 0
                            current_utterance_audio = np.empty(0, dtype=np.float32)

                            await websocket.send_text(
                                StreamingTranscriptChunk(
                                    event="speech_end",
                                    start=round(end_sec, 3),
                                    text=""
                                ).model_dump_json()
                            )

    except WebSocketDisconnect:
        logger.info("WebSocket streaming client disconnected normally.")
    except Exception as e:
        logger.error("Unexpected error in WebSocket streaming loop: %s", e)
        try:
            await websocket.send_text(
                StreamingTranscriptChunk(
                    event="error",
                    text=f"Server streaming error: {str(e)}"
                ).model_dump_json()
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass

