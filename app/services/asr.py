"""
VoxSplit Automatic Speech Recognition (ASR) Service.
High-throughput Whisper inference engine powered by faster-whisper (CTranslate2),
featuring quantized int8 CPU execution, lazy thread-safe initialization, and word-level alignment.
"""

import threading
import logging
from typing import List, Optional, Tuple
import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

from app.core.config import Settings, get_settings
from app.models.schemas import WordTiming

logger = logging.getLogger("voxsplit.asr")


class ASRService:
    """
    Singleton-capable speech recognition service encapsulating faster-whisper.
    Employs int8 quantization for ultra-fast local CPU inference.
    """

    _instance: Optional["ASRService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: Optional[Settings] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.settings = settings or get_settings()
        self.model_size = self.settings.WHISPER_MODEL_SIZE
        self.device = self.settings.WHISPER_DEVICE
        self.compute_type = self.settings.WHISPER_COMPUTE_TYPE
        self.beam_size = self.settings.WHISPER_BEAM_SIZE
        self.cpu_threads = self.settings.WHISPER_CPU_THREADS
        self.word_timestamps = self.settings.WHISPER_WORD_TIMESTAMPS
        self.default_language = self.settings.WHISPER_LANGUAGE

        self._model: Optional["WhisperModel"] = None
        self._load_lock = threading.Lock()
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Checks if the faster-whisper CTranslate2 model is loaded into memory."""
        return self._model is not None

    def load_model(self) -> None:
        """
        Thread-safe explicit loader for the Whisper model.
        Called on service startup or lazily upon first inference invocation.
        """
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            if not FASTER_WHISPER_AVAILABLE:
                raise RuntimeError(
                    "faster-whisper is not installed. Please install faster-whisper to execute ASRService."
                )

            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s, cpu_threads=%d)...",
                self.model_size,
                self.device,
                self.compute_type,
                self.cpu_threads
            )

            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads
                )
                logger.info("faster-whisper model '%s' successfully loaded into memory.", self.model_size)
            except Exception as e:
                logger.error("Failed to load faster-whisper model: %s", str(e))
                raise RuntimeError(f"Whisper initialization failure: {e}") from e

    def get_model(self) -> "WhisperModel":
        """Returns the loaded model instance, instantiating it lazily if not already present."""
        if self._model is None:
            self.load_model()
        return self._model

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, float, List[WordTiming], Optional[str]]:
        """
        Transcribes a float32 16kHz mono audio slice using the faster-whisper engine.

        Args:
            audio_chunk: 1D float32 NumPy array normalized to [-1.0, 1.0].
            language: Optional ISO-639-1 language code (e.g. 'en', 'es'). Defaults to auto-detect.

        Returns:
            Tuple of (text, average_confidence, word_timings, detected_language).
        """
        if len(audio_chunk) == 0:
            return "", 1.0, [], language

        model = self.get_model()
        lang = language or self.default_language

        # Note: We disable internal vad_filter since VoxSplit's dedicated VAD already segmented the audio
        segments, info = model.transcribe(
            audio=audio_chunk,
            beam_size=self.beam_size,
            language=lang,
            word_timestamps=self.word_timestamps,
            vad_filter=False,
            temperature=0.0
        )

        text_parts: List[str] = []
        word_timings: List[WordTiming] = []
        confidence_scores: List[float] = []

        detected_lang = info.language if info else lang

        for seg in segments:
            clean_text = seg.text.strip()
            if clean_text:
                text_parts.append(clean_text)

            # Segment-level probability approximation
            seg_conf = np.exp(seg.avg_logprob) if hasattr(seg, "avg_logprob") else 0.95
            confidence_scores.append(float(np.clip(seg_conf, 0.0, 1.0)))

            # Collect word-level timestamps if available
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    word_timings.append(
                        WordTiming(
                            word=w.word.strip(),
                            start=round(float(w.start), 3),
                            end=round(float(w.end), 3),
                            probability=round(float(w.probability), 3) if hasattr(w, "probability") else 1.0
                        )
                    )

        full_text = " ".join(text_parts).strip()
        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 1.0

        return full_text, round(avg_confidence, 3), word_timings, detected_lang

