"""
VoxSplit Voice Activity Detection (VAD) Service.
High-precision voiced segment detection combining WebRTC VAD with energy/ZCR gating
and hysteresis hangover smoothing to eliminate chunk boundary artifacts.
"""

import collections
import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from app.core.config import Settings, get_settings
from app.core.audio_utils import (
    float_to_int16,
    compute_frame_energy_dbfs,
    compute_zero_crossing_rate,
)

logger = logging.getLogger("voxsplit.vad")


class SpeechInterval:
    """Represents a validated speech region in seconds."""
    def __init__(self, start: float, end: float):
        self.start = float(start)
        self.end = float(end)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_tuple(self) -> Tuple[float, float]:
        return (round(self.start, 3), round(self.end, 3))

    def __repr__(self) -> str:
        return f"SpeechInterval(start={self.start:.3f}, end={self.end:.3f}, duration={self.duration:.3f})"


class VoiceActivityDetector:
    """
    Production-grade Voice Activity Detector.
    Integrates WebRTC VAD with spectral energy gating and ring-buffer smoothing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.sample_rate = self.settings.SAMPLE_RATE
        self.frame_duration_ms = self.settings.VAD_FRAME_DURATION_MS
        self.aggressiveness = self.settings.VAD_AGGRESSIVENESS
        self.padding_duration_ms = self.settings.VAD_PADDING_DURATION_MS
        self.min_speech_duration_ms = self.settings.VAD_MIN_SPEECH_DURATION_MS
        self.max_speech_duration_sec = self.settings.VAD_MAX_SPEECH_DURATION_SEC
        self.energy_threshold_db = self.settings.VAD_ENERGY_THRESHOLD_DB

        # Samples per analysis frame (e.g. 16000 * 30 / 1000 = 480 samples)
        self.frame_length_samples = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.frame_bytes_length = self.frame_length_samples * 2  # 16-bit PCM = 2 bytes/sample

        # Initialize WebRTC VAD
        if WEBRTC_AVAILABLE:
            try:
                self._vad = webrtcvad.Vad(self.aggressiveness)
            except Exception as e:
                logger.warning("Could not initialize webrtcvad (%s), falling back to energy VAD", e)
                self._vad = None
        else:
            logger.warning("webrtcvad not installed, operating in native energy/ZCR mode")
            self._vad = None

    def _is_frame_voiced(self, frame_bytes: bytes, frame_float: np.ndarray) -> bool:
        """
        Determines whether an individual frame contains active speech using
        WebRTC VAD when available, reinforced by an energy floor gate.
        """
        # Calculate RMS energy of this frame
        rms = np.sqrt(np.mean(frame_float ** 2) + 1e-9)
        energy_db = 20.0 * np.log10(rms)

        # Hard silence floor rejection
        if energy_db < self.energy_threshold_db:
            return False

        if self._vad is not None:
            try:
                return self._vad.is_speech(frame_bytes, self.sample_rate)
            except Exception as e:
                logger.debug("WebRTC frame evaluation error: %s", e)

        # Fallback heuristic: Zero-crossing rate and energy
        # Voiced speech typically has moderate ZCR and higher energy
        zcr = np.mean(np.abs(np.diff(np.sign(frame_float)))) / 2.0
        return bool(energy_db >= (self.energy_threshold_db + 6.0) and (0.01 <= zcr <= 0.45))

    def detect_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        Segments an audio array into contiguous voiced intervals (start_sec, end_sec).

        Args:
            audio: 1D float32 NumPy array normalized to [-1.0, 1.0].
            sample_rate: Audio sampling rate (must match self.sample_rate).

        Returns:
            List of (start_sec, end_sec) tuples rounded to milliseconds.
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(
                f"Audio sample rate ({sample_rate} Hz) must match detector rate ({self.sample_rate} Hz)."
            )

        total_samples = len(audio)
        if total_samples < self.frame_length_samples:
            # Too short for frame analysis; if audio has energy, return whole slice
            if compute_frame_energy_dbfs(audio)[0] > self.energy_threshold_db:
                return [(0.0, round(total_samples / self.sample_rate, 3))]
            return []

        # Convert full audio to int16 PCM array for WebRTC VAD
        int16_audio = float_to_int16(audio)

        # Sliding window parameters
        num_padding_frames = max(1, int(self.padding_duration_ms / self.frame_duration_ms))
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_intervals: List[SpeechInterval] = []

        start_time_sec = 0.0
        last_voiced_time_sec = 0.0
        current_time_sec = 0.0
        frame_sec = self.frame_duration_ms / 1000.0

        num_frames = total_samples // self.frame_length_samples

        for i in range(num_frames):
            idx_start = i * self.frame_length_samples
            idx_end = idx_start + self.frame_length_samples

            frame_pcm = int16_audio[idx_start:idx_end].tobytes()
            frame_float = audio[idx_start:idx_end]
            current_time_sec = i * frame_sec

            is_speech = self._is_frame_voiced(frame_pcm, frame_float)

            if not triggered:
                ring_buffer.append((current_time_sec, is_speech))
                num_voiced = sum(1 for _, speech in ring_buffer if speech)

                # Trigger speech onset when 80% of window is voiced
                if num_voiced >= 0.8 * ring_buffer.maxlen:
                    triggered = True
                    # Set start time to the first voiced frame in the buffer
                    voiced_frames_in_buf = [t for t, speech in ring_buffer if speech]
                    start_time_sec = voiced_frames_in_buf[0] if voiced_frames_in_buf else ring_buffer[0][0]
                    last_voiced_time_sec = current_time_sec + frame_sec
                    ring_buffer.clear()
            else:
                if is_speech:
                    last_voiced_time_sec = current_time_sec + frame_sec

                ring_buffer.append((current_time_sec, is_speech))
                num_unvoiced = sum(1 for _, speech in ring_buffer if not speech)

                # Check if speech should end (silence hangover expired)
                hangover_expired = (num_unvoiced >= 0.85 * ring_buffer.maxlen)

                # Check if current segment exceeded max allowed speech chunk duration
                duration_exceeded = (current_time_sec - start_time_sec) >= self.max_speech_duration_sec

                if hangover_expired or duration_exceeded:
                    end_time_sec = min(last_voiced_time_sec, current_time_sec + frame_sec)
                    interval = SpeechInterval(start=start_time_sec, end=end_time_sec)

                    if interval.duration >= (self.min_speech_duration_ms / 1000.0):
                        voiced_intervals.append(interval)

                    triggered = False
                    ring_buffer.clear()

        # Flush any open trailing speech segment
        if triggered:
            final_end = min(total_samples / self.sample_rate, last_voiced_time_sec)
            interval = SpeechInterval(start=start_time_sec, end=final_end)
            if interval.duration >= (self.min_speech_duration_ms / 1000.0):
                voiced_intervals.append(interval)

        # Merge segments with tiny micro-gap (< 150ms) to prevent fragmented words
        merged_intervals = self._merge_adjacent_intervals(voiced_intervals, max_gap_sec=0.15)

        # Split any oversized segments (> max_speech_duration_sec) at local energy minima
        final_intervals: List[SpeechInterval] = []
        for interval in merged_intervals:
            if interval.duration > self.max_speech_duration_sec:
                splits = self._split_long_interval(audio, interval)
                final_intervals.extend(splits)
            else:
                final_intervals.append(interval)

        return [iv.to_tuple() for iv in final_intervals]

    def _merge_adjacent_intervals(
        self,
        intervals: List[SpeechInterval],
        max_gap_sec: float = 0.25
    ) -> List[SpeechInterval]:
        """Merges consecutive speech chunks separated by less than max_gap_sec silence."""
        if not intervals:
            return []

        merged: List[SpeechInterval] = [intervals[0]]
        for curr in intervals[1:]:
            prev = merged[-1]
            gap = curr.start - prev.end
            if gap <= max_gap_sec and (curr.end - prev.start) <= self.max_speech_duration_sec:
                prev.end = curr.end
            else:
                merged.append(curr)
        return merged

    def _split_long_interval(
        self,
        audio: np.ndarray,
        interval: SpeechInterval
    ) -> List[SpeechInterval]:
        """
        Splits a long interval exceeding max_speech_duration_sec at the point
        of lowest energy within the middle third of the segment.
        """
        splits: List[SpeechInterval] = []
        start = interval.start
        end = interval.end

        while (end - start) > self.max_speech_duration_sec:
            target_split = start + self.max_speech_duration_sec
            # Search window: between (target_split - 5.0) and target_split
            search_start_sec = max(start + 5.0, target_split - 6.0)
            search_end_sec = target_split

            idx_s = int(search_start_sec * self.sample_rate)
            idx_e = int(search_end_sec * self.sample_rate)
            sub_audio = audio[idx_s:idx_e]

            if len(sub_audio) > self.frame_length_samples:
                # Find frame with minimum RMS
                hop = self.frame_length_samples
                rms_values = [
                    np.mean(sub_audio[k:k+hop] ** 2)
                    for k in range(0, len(sub_audio) - hop, hop)
                ]
                if rms_values:
                    min_idx = int(np.argmin(rms_values))
                    split_sec = search_start_sec + (min_idx * (self.frame_duration_ms / 1000.0))
                else:
                    split_sec = target_split - 1.0
            else:
                split_sec = target_split - 1.0

            splits.append(SpeechInterval(start=start, end=split_sec))
            start = split_sec

        if (end - start) >= (self.min_speech_duration_ms / 1000.0):
            splits.append(SpeechInterval(start=start, end=end))

        return splits
