"""
VoxSplit Speaker Diarization Engine.
Acoustic feature extraction via spectral MFCCs and delta dynamics, paired with
cosine similarity clustering and online speaker profile tracking.
"""

import collections
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering

from app.core.config import Settings, get_settings
from app.models.schemas import DiarizationSegment

logger = logging.getLogger("voxsplit.diarizer")


class SpeakerProfile:
    """Tracks running acoustic signature and metadata for an identified speaker."""

    def __init__(self, speaker_id: str, initial_embedding: np.ndarray):
        self.speaker_id = speaker_id
        # L2-normalized centroid vector
        self.centroid = initial_embedding / (np.linalg.norm(initial_embedding) + 1e-9)
        self.utterance_count = 1
        self.total_duration_sec = 0.0
        # Rolling exemplar buffer for intra-speaker variance comparison
        self.exemplars: collections.deque = collections.deque(maxlen=10)
        self.exemplars.append(self.centroid.copy())

    def update(self, new_embedding: np.ndarray, duration_sec: float, momentum: float = 0.85) -> None:
        """
        Updates the running centroid using exponential moving average and normalizes to unit sphere.
        """
        norm_emb = new_embedding / (np.linalg.norm(new_embedding) + 1e-9)
        self.centroid = (momentum * self.centroid) + ((1.0 - momentum) * norm_emb)
        self.centroid = self.centroid / (np.linalg.norm(self.centroid) + 1e-9)
        self.utterance_count += 1
        self.total_duration_sec += duration_sec
        self.exemplars.append(norm_emb)

    def similarity(self, embedding: np.ndarray) -> float:
        """Computes cosine similarity between input embedding and speaker profile."""
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-9)
        centroid_sim = float(np.dot(self.centroid, norm_emb))

        # Check maximum exemplar similarity for robustness to pitch/intonation shifts
        exemplar_sims = [float(np.dot(ex, norm_emb)) for ex in self.exemplars]
        max_exemplar_sim = max(exemplar_sims) if exemplar_sims else centroid_sim

        # Weighted blend between centroid and closest exemplar
        return float(0.6 * centroid_sim + 0.4 * max_exemplar_sim)


class SpeakerDiarizer:
    """
    Modular speaker diarization pipeline supporting both online sequential tracking
    and offline global agglomerative clustering.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.sample_rate = self.settings.SAMPLE_RATE
        self.similarity_threshold = self.settings.SPEAKER_SIMILARITY_THRESHOLD
        self.window_sec = self.settings.DIARIZATION_WINDOW_SEC
        self.hop_sec = self.settings.DIARIZATION_HOP_SEC
        self.n_mfcc = self.settings.N_MFCC
        self.max_speakers = self.settings.MAX_SPEAKERS

        # Dynamic online speaker directory
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._next_speaker_idx = 0

    def reset(self) -> None:
        """Resets active speaker clusters (e.g. between distinct audio sessions)."""
        self._profiles.clear()
        self._next_speaker_idx = 0

    def extract_embedding(self, audio_slice: np.ndarray) -> np.ndarray:
        """
        Extracts a compact, rotation-invariant speaker acoustic embedding vector from a waveform.
        Combines timbre-characterizing MFCCs (excluding energy coeff 0), delta dynamics,
        pitch-sensitive chroma features, and spectral contrast statistics.
        """
        if len(audio_slice) < int(self.sample_rate * 0.1):  # Less than 100ms
            pad_len = int(self.sample_rate * 0.1) - len(audio_slice)
            audio_slice = np.pad(audio_slice, (0, pad_len), mode="constant")

        # Framing parameters
        n_fft = min(1024, len(audio_slice))
        hop_length = n_fft // 2

        try:
            # 1. Mel-Frequency Cepstral Coefficients (1..N, skipping 0 which is just energy)
            mfcc_full = librosa.feature.mfcc(
                y=audio_slice,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc + 1,
                n_fft=n_fft,
                hop_length=hop_length
            )
            mfcc = mfcc_full[1:]  # Exclude energy component

            # 2. Delta dynamics
            mfcc_delta = librosa.feature.delta(mfcc, order=1)

            # 3. Harmonic pitch / chroma distribution (captures F0 / vocal fold vibration rate)
            chroma = librosa.feature.chroma_stft(
                y=audio_slice, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_chroma=12
            )

            # 4. Spectral contrast across sub-bands (captures formant prominence and vocal tract resonance)
            contrast = librosa.feature.spectral_contrast(
                y=audio_slice, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length
            )

            # Statistical aggregation and sub-band L2 normalization
            v_mfcc = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
            v_mfcc = v_mfcc / (np.linalg.norm(v_mfcc) + 1e-9)

            v_delta = np.concatenate([np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1)])
            v_delta = v_delta / (np.linalg.norm(v_delta) + 1e-9)

            v_chroma = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
            v_chroma = v_chroma / (np.linalg.norm(v_chroma) + 1e-9)

            v_contrast = np.concatenate([np.mean(contrast, axis=1), np.std(contrast, axis=1)])
            v_contrast = v_contrast / (np.linalg.norm(v_contrast) + 1e-9)

            features = np.concatenate([v_mfcc, v_delta, v_chroma, v_contrast])

        except Exception as e:
            logger.warning("Feature extraction exception: %s. Using fallback FFT spectral shape.", e)
            fft_mag = np.abs(np.fft.rfft(audio_slice[:1024]))
            features = np.concatenate([fft_mag[:64], np.diff(fft_mag[:65])])

        # L2-normalize overall embedding to unit hypersphere
        norm = np.linalg.norm(features)
        if norm > 1e-9:
            features = features / norm
        else:
            features = np.zeros_like(features)

        return features.astype(np.float32)

    def assign_speaker_online(
        self,
        audio_slice: np.ndarray,
        duration_sec: float
    ) -> Tuple[str, float]:
        """
        Assigns an incoming speech segment to an existing speaker profile or registers
        a new speaker identity using online cosine similarity matching.

        Returns:
            Tuple of (speaker_id, similarity_score).
        """
        embedding = self.extract_embedding(audio_slice)

        if not self._profiles:
            speaker_id = f"SPEAKER_{self._next_speaker_idx:02d}"
            self._next_speaker_idx += 1
            self._profiles[speaker_id] = SpeakerProfile(speaker_id, embedding)
            return speaker_id, 1.0

        # Compare with known speakers
        best_speaker = None
        best_sim = -1.0

        for spk_id, profile in self._profiles.items():
            sim = profile.similarity(embedding)
            if sim > best_sim:
                best_sim = sim
                best_speaker = spk_id

        # Check against matching threshold
        if best_sim >= self.similarity_threshold:
            self._profiles[best_speaker].update(embedding, duration_sec)
            return best_speaker, round(best_sim, 3)

        # New speaker discovery (if under ceiling limit)
        if len(self._profiles) < self.max_speakers:
            speaker_id = f"SPEAKER_{self._next_speaker_idx:02d}"
            self._next_speaker_idx += 1
            self._profiles[speaker_id] = SpeakerProfile(speaker_id, embedding)
            return speaker_id, 1.0
        else:
            # Constrained capacity: fall back to closest existing cluster
            self._profiles[best_speaker].update(embedding, duration_sec)
            return best_speaker, round(best_sim, 3)

    def diarize_segments(
        self,
        audio: np.ndarray,
        segments: List[Tuple[float, float]]
    ) -> List[DiarizationSegment]:
        """
        Diarizes a sequence of time-bounded speech segments.
        Utilizes global agglomerative clustering with cosine distance when multiple
        segments are present to maximize cross-conversation consistency.
        """
        if not segments:
            return []

        # If only 1 segment, immediately assign to SPEAKER_00
        if len(segments) == 1:
            start, end = segments[0]
            return [DiarizationSegment(speaker="SPEAKER_00", start=start, end=end, confidence=1.0)]

        # Extract embeddings for all speech intervals
        embeddings: List[np.ndarray] = []
        valid_intervals: List[Tuple[float, float]] = []

        for start, end in segments:
            s_idx = max(0, int(round(start * self.sample_rate)))
            e_idx = min(len(audio), int(round(end * self.sample_rate)))
            chunk = audio[s_idx:e_idx]

            if len(chunk) > 0:
                emb = self.extract_embedding(chunk)
                embeddings.append(emb)
                valid_intervals.append((start, end))

        if not embeddings:
            return []

        emb_matrix = np.array(embeddings)  # Shape: (N, D)

        # If only 2-3 segments or hierarchical clustering is not feasible, use online sequential
        if len(embeddings) < 3:
            self.reset()
            results: List[DiarizationSegment] = []
            for (start, end), emb in zip(valid_intervals, embeddings):
                s_idx = int(start * self.sample_rate)
                e_idx = int(end * self.sample_rate)
                spk, conf = self.assign_speaker_online(audio[s_idx:e_idx], end - start)
                results.append(DiarizationSegment(speaker=spk, start=start, end=end, confidence=conf))
            return results

        # Global Agglomerative Clustering with Cosine Distance
        distance_threshold = 1.0 - self.similarity_threshold
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold
            )
            labels = clustering.fit_predict(emb_matrix)
        except Exception as e:
            logger.warning("Agglomerative clustering failed: %s, falling back to online tracking", e)
            self.reset()
            results = []
            for (start, end), emb in zip(valid_intervals, embeddings):
                s_idx = int(start * self.sample_rate)
                e_idx = int(end * self.sample_rate)
                spk, conf = self.assign_speaker_online(audio[s_idx:e_idx], end - start)
                results.append(DiarizationSegment(speaker=spk, start=start, end=end, confidence=conf))
            return results

        # Map cluster labels to chronological appearance (SPEAKER_00, SPEAKER_01, ...)
        label_to_speaker_id: Dict[int, str] = {}
        speaker_counter = 0

        diarized_segments: List[DiarizationSegment] = []
        for (start, end), label, emb in zip(valid_intervals, labels, embeddings):
            if label not in label_to_speaker_id:
                label_to_speaker_id[label] = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1

            speaker_id = label_to_speaker_id[label]
            diarized_segments.append(
                DiarizationSegment(
                    speaker=speaker_id,
                    start=start,
                    end=end,
                    confidence=0.95
                )
            )

        return diarized_segments
