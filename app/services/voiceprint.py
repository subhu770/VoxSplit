"""
VoxSplit Speaker Voiceprint Enrollment & Identification Service.
Enables registration of known speaker voiceprints and automatic identity
re-identification across audio recordings.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.config import Settings, get_settings
from app.services.diarizer import SpeakerDiarizer

logger = logging.getLogger("voxsplit.voiceprint")


class VoiceprintService:
    """
    Manages known speaker profiles, embedding serialization, and identity matching.
    """

    _instance: Optional["VoiceprintService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Optional[Settings] = None, storage_path: str = "data/voiceprints.json"):
        if getattr(self, "_initialized", False):
            return

        self.settings = settings or get_settings()
        self.storage_path = storage_path
        self._diarizer = SpeakerDiarizer(self.settings)
        # Dictionary mapping speaker_name -> L2-normalized numpy embedding vector
        self._voiceprints: Dict[str, np.ndarray] = {}
        self._similarity_threshold = self.settings.SPEAKER_SIMILARITY_THRESHOLD

        self._load_from_disk()
        self._initialized = True

    def _load_from_disk(self) -> None:
        """Loads enrolled voiceprints from disk if file exists."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, emb_list in data.items():
                emb = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-9:
                    self._voiceprints[name] = emb / norm
            logger.info("Loaded %d enrolled voiceprints from %s", len(self._voiceprints), self.storage_path)
        except Exception as e:
            logger.warning("Could not read voiceprints from disk (%s): %s", self.storage_path, e)

    def _save_to_disk(self) -> None:
        """Persists current voiceprint bank to disk as JSON."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            serializable = {
                name: emb.tolist()
                for name, emb in self._voiceprints.items()
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            logger.info("Saved %d enrolled voiceprints to %s", len(self._voiceprints), self.storage_path)
        except Exception as e:
            logger.error("Failed to save voiceprints to disk: %s", e)

    def enroll_speaker(self, speaker_name: str, audio: np.ndarray) -> Tuple[str, int]:
        """
        Enrolls a new speaker identity from reference audio waveform.

        Args:
            speaker_name: Unique human-readable speaker identifier (e.g. 'Alice').
            audio: 1D float32 audio array at 16,000 Hz.

        Returns:
            Tuple of (speaker_name, embedding_dimension).
        """
        clean_name = speaker_name.strip()
        if not clean_name:
            raise ValueError("Speaker name cannot be empty.")

        embedding = self._diarizer.extract_embedding(audio)
        self._voiceprints[clean_name] = embedding
        self._save_to_disk()

        logger.info("Enrolled speaker '%s' with %d-dimensional embedding.", clean_name, len(embedding))
        return clean_name, len(embedding)

    def identify_speaker(self, audio_slice: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Matches an audio slice against enrolled voiceprints.

        Returns:
            Tuple of (matched_name, similarity) if similarity >= threshold, else None.
        """
        if not self._voiceprints:
            return None

        embedding = self._diarizer.extract_embedding(audio_slice)
        best_name = None
        best_sim = -1.0

        for name, enrolled_emb in self._voiceprints.items():
            sim = float(np.dot(embedding, enrolled_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self._similarity_threshold:
            return best_name, round(best_sim, 3)

        return None

    def list_enrolled_speakers(self) -> List[str]:
        """Returns sorted list of currently registered speaker names."""
        return sorted(list(self._voiceprints.keys()))

    def delete_speaker(self, speaker_name: str) -> bool:
        """Removes a speaker profile from memory and disk."""
        if speaker_name in self._voiceprints:
            del self._voiceprints[speaker_name]
            self._save_to_disk()
            return True
        return False

    def clear(self) -> None:
        """Removes all enrolled speaker profiles."""
        self._voiceprints.clear()
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except OSError:
                pass
