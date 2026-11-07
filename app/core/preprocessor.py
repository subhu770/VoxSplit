"""
VoxSplit Audio Preprocessing & Speech Conditioning Module.
Provides low-latency acoustic filtering, DC offset removal, high-pass rumble rejection,
pre-emphasis filtering, and chunk boundary cross-fading to eliminate acoustic artifacts.
"""

import numpy as np
from scipy import signal


class AudioPreprocessor:
    """
    Applies production DSP conditioning to raw audio before VAD and ASR.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        highpass_cutoff_hz: float = 80.0,
        pre_emphasis_coeff: float = 0.97
    ):
        self.sample_rate = sample_rate
        self.highpass_cutoff_hz = highpass_cutoff_hz
        self.pre_emphasis_coeff = pre_emphasis_coeff

        # Design 4th order Butterworth high-pass filter (b, a coefficients)
        nyquist = 0.5 * self.sample_rate
        norm_cutoff = self.highpass_cutoff_hz / nyquist
        self._b, self._a = signal.butter(4, norm_cutoff, btype="highpass")

    def remove_dc_and_rumble(self, audio: np.ndarray) -> np.ndarray:
        """
        Removes DC bias and low-frequency mechanical rumble (< 80 Hz)
        using zero-phase forward-backward Butterworth filtering.
        """
        if len(audio) < 16:
            return audio - np.mean(audio)

        # Remove pure DC offset first
        audio_zero_mean = audio - np.mean(audio)

        # Forward-backward filter to prevent phase distortion
        filtered = signal.filtfilt(self._b, self._a, audio_zero_mean)
        return filtered.astype(np.float32)

    def apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies pre-emphasis filter: y[t] = x[t] - alpha * x[t-1].
        Amplifies high-frequency formant energy, improving phoneme resolution for Whisper.
        """
        if len(audio) == 0:
            return audio
        emphasized = np.append(audio[0], audio[1:] - self.pre_emphasis_coeff * audio[:-1])
        return emphasized.astype(np.float32)

    def apply_boundary_crossfade(
        self,
        chunk_a: np.ndarray,
        chunk_b: np.ndarray,
        crossfade_ms: float = 10.0
    ) -> np.ndarray:
        """
        Smooths the transition between two adjacent audio chunks using an equal-power
        cross-fade window to eliminate boundary clicks and pops.
        """
        crossfade_samples = int(self.sample_rate * (crossfade_ms / 1000.0))
        if len(chunk_a) < crossfade_samples or len(chunk_b) < crossfade_samples or crossfade_samples == 0:
            return np.concatenate([chunk_a, chunk_b])

        t = np.linspace(0.0, 1.0, crossfade_samples, endpoint=True)
        # Equal-power sine/cosine crossfade curves
        fade_out = np.cos(0.5 * np.pi * t)
        fade_in = np.sin(0.5 * np.pi * t)

        overlap = (chunk_a[-crossfade_samples:] * fade_out) + (chunk_b[:crossfade_samples] * fade_in)

        return np.concatenate([
            chunk_a[:-crossfade_samples],
            overlap,
            chunk_b[crossfade_samples:]
        ]).astype(np.float32)

    def process(self, audio: np.ndarray, pre_emphasize: bool = False) -> np.ndarray:
        """
        Executes full standard preprocessing pipeline:
        1. DC offset removal & highpass rumble rejection.
        2. Optional pre-emphasis filtering.
        3. Peak clipping protection.
        """
        if len(audio) == 0:
            return audio

        conditioned = self.remove_dc_and_rumble(audio)

        if pre_emphasize:
            conditioned = self.apply_pre_emphasis(conditioned)

        # Safe clamp to [-1.0, 1.0]
        max_val = np.max(np.abs(conditioned))
        if max_val > 1.0:
            conditioned = conditioned / max_val

        return conditioned.astype(np.float32)
