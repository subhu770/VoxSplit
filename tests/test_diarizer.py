"""
Unit Tests for VoxSplit Speaker Diarization Engine.
"""

import numpy as np
import pytest

from app.core.config import Settings
from app.services.diarizer import SpeakerDiarizer, SpeakerProfile
from samples.generate_synthetic_sample import generate_harmonic_voice


def test_embedding_extraction_dimension_and_norm(synthetic_speaker_a):
    """Verifies that extracted embeddings are non-empty, 1D float32, and strictly L2-normalized."""
    diarizer = SpeakerDiarizer()
    emb = diarizer.extract_embedding(synthetic_speaker_a)

    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) > 10
    norm = float(np.linalg.norm(emb))
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_differentiates_speakers(
    synthetic_speaker_a,
    synthetic_speaker_b,
    sample_rate
):
    """
    Validates that intra-speaker similarity (same F0/timbre) is significantly
    higher than inter-speaker similarity (different F0/timbre).
    """
    # Second distinct utterance from Speaker A
    synthetic_speaker_a2 = generate_harmonic_voice(
        duration_sec=2.0,
        fundamental_hz=130.0,
        formant_peaks=[700.0, 1220.0, 2600.0],
        sample_rate=sample_rate,
        amplitude=0.55
    )

    diarizer = SpeakerDiarizer()
    emb_a1 = diarizer.extract_embedding(synthetic_speaker_a)
    emb_a2 = diarizer.extract_embedding(synthetic_speaker_a2)
    emb_b = diarizer.extract_embedding(synthetic_speaker_b)

    intra_sim = float(np.dot(emb_a1, emb_a2))
    inter_sim = float(np.dot(emb_a1, emb_b))

    assert intra_sim > 0.85
    assert intra_sim > inter_sim
    assert (intra_sim - inter_sim) > 0.15


def test_online_speaker_assignment_consistency(
    synthetic_speaker_a,
    synthetic_speaker_b,
    sample_rate
):
    """
    Simulates consecutive speech turns:
    Turn 1: Speaker A -> SPEAKER_00
    Turn 2: Speaker B -> SPEAKER_01
    Turn 3: Speaker A -> Must match back to SPEAKER_00
    """
    settings = Settings(SPEAKER_SIMILARITY_THRESHOLD=0.75)
    diarizer = SpeakerDiarizer(settings=settings)
    diarizer.reset()

    # Second sample of Speaker A
    speaker_a_again = generate_harmonic_voice(
        duration_sec=1.8,
        fundamental_hz=130.0,
        formant_peaks=[700.0, 1220.0, 2600.0],
        sample_rate=sample_rate,
        amplitude=0.6
    )

    spk1, conf1 = diarizer.assign_speaker_online(synthetic_speaker_a, 2.0)
    assert spk1 == "SPEAKER_00"

    spk2, conf2 = diarizer.assign_speaker_online(synthetic_speaker_b, 2.0)
    assert spk2 == "SPEAKER_01"

    spk3, conf3 = diarizer.assign_speaker_online(speaker_a_again, 1.8)
    assert spk3 == "SPEAKER_00"


def test_batch_diarize_segments(sample_rate):
    """Verifies batch segment diarization returns DiarizationSegment objects with consistent identities."""
    voice_a = generate_harmonic_voice(1.5, 120.0, [650.0, 1100.0], sample_rate)
    voice_b = generate_harmonic_voice(1.5, 250.0, [450.0, 1800.0], sample_rate)
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)

    # Audio sequence: A (0-1.5), silence (1.5-2.0), B (2.0-3.5), silence (3.5-4.0), A (4.0-5.5)
    audio = np.concatenate([voice_a, silence, voice_b, silence, voice_a])
    intervals = [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]

    diarizer = SpeakerDiarizer()
    results = diarizer.diarize_segments(audio, intervals)

    assert len(results) == 3
    assert results[0].speaker == results[2].speaker
    assert results[0].speaker != results[1].speaker
    assert results[0].start == 0.0
    assert results[1].start == 2.0
    assert results[2].start == 4.0


def test_speaker_profile_update():
    """Checks profile centroid updating and momentum weighting."""
    emb1 = np.ones(10, dtype=np.float32)
    profile = SpeakerProfile("SPEAKER_00", emb1)
    assert profile.utterance_count == 1

    emb2 = np.ones(10, dtype=np.float32) * 2.0
    profile.update(emb2, duration_sec=1.5, momentum=0.5)
    assert profile.utterance_count == 2
    assert profile.total_duration_sec == 1.5
    assert len(profile.exemplars) == 2

