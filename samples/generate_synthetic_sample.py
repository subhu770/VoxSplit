"""
VoxSplit Synthetic Multi-Speaker Audio Generator.
Creates a multi-speaker, multi-frequency synthetic WAV file with distinct pitch fundamentals,
harmonic formants, and silence intervals to benchmark VAD and Diarization logic.
"""

import os
import argparse
import numpy as np
import soundfile as sf


def generate_harmonic_voice(
    duration_sec: float,
    fundamental_hz: float,
    formant_peaks: list[float],
    sample_rate: int = 16000,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    Synthesizes a vowel-like vocal sound with a rich harmonic overtone series
    filtered through synthetic vocal tract formant resonances.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

    # Sum fundamental and overtones up to Nyquist
    waveform = np.zeros(num_samples, dtype=np.float32)
    max_harmonic = int((sample_rate / 2.0) / fundamental_hz)

    for h in range(1, min(max_harmonic, 25)):
        freq = h * fundamental_hz
        # Formant weighting filter (approximate acoustic resonance response)
        gain = 1.0 / h
        for f_peak in formant_peaks:
            dist = abs(freq - f_peak)
            bandwidth = 120.0
            gain += 1.8 * np.exp(-0.5 * ((dist / bandwidth) ** 2))

        waveform += gain * np.sin(2.0 * np.pi * freq * t)

    # Smooth trapezoidal amplitude envelope (15ms attack and decay)
    ramp_len = min(int(0.015 * sample_rate), num_samples // 4)
    envelope = np.ones(num_samples, dtype=np.float32)
    if ramp_len > 0:
        envelope[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        envelope[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)

    # Subtle vibrato (5 Hz pitch oscillation)
    vibrato = 1.0 + 0.03 * np.sin(2.0 * np.pi * 5.0 * t)
    voiced = waveform * envelope * vibrato

    # Normalize amplitude
    peak = np.max(np.abs(voiced))
    if peak > 0:
        voiced = (voiced / peak) * amplitude

    return voiced.astype(np.float32)


def generate_synthetic_dialogue(
    output_path: str = "samples/multi_speaker_test.wav",
    sample_rate: int = 16000
) -> dict:
    """
    Builds an interleaved multi-speaker conversation:
    - Speaker 0 (Male timbre: F0 ~ 130 Hz, formants [700, 1220, 2600] Hz)
    - Speaker 1 (Female timbre: F0 ~ 240 Hz, formants [400, 1900, 2900] Hz)
    Separated by controlled silence gaps.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    turns = [
        # (speaker_id, duration_sec, silence_after_sec, f0, formants)
        ("SPEAKER_00", 1.8, 0.5, 130.0, [700.0, 1220.0, 2600.0]),
        ("SPEAKER_01", 1.6, 0.4, 240.0, [400.0, 1900.0, 2900.0]),
        ("SPEAKER_00", 2.2, 0.6, 130.0, [700.0, 1220.0, 2600.0]),
        ("SPEAKER_01", 1.9, 0.5, 240.0, [400.0, 1900.0, 2900.0]),
    ]

    timeline = []
    current_time = 0.0
    audio_segments = []

    # Initial brief silence
    initial_silence_samples = int(0.3 * sample_rate)
    audio_segments.append(np.zeros(initial_silence_samples, dtype=np.float32))
    current_time += 0.3

    for spk_id, dur, sil_after, f0, formants in turns:
        voiced_chunk = generate_harmonic_voice(
            duration_sec=dur,
            fundamental_hz=f0,
            formant_peaks=formants,
            sample_rate=sample_rate,
            amplitude=0.65
        )
        audio_segments.append(voiced_chunk)

        seg_start = current_time
        seg_end = current_time + dur
        timeline.append({
            "speaker": spk_id,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(dur, 3),
            "f0": f0
        })

        current_time += dur

        # Append silence gap
        sil_samples = int(sil_after * sample_rate)
        if sil_samples > 0:
            audio_segments.append(np.zeros(sil_samples, dtype=np.float32))
            current_time += sil_after

    full_audio = np.concatenate(audio_segments).astype(np.float32)

    # Write 16-bit PCM WAV
    sf.write(output_path, full_audio, sample_rate, subtype="PCM_16")

    ground_truth = {
        "output_path": output_path,
        "sample_rate": sample_rate,
        "total_duration": round(len(full_audio) / sample_rate, 3),
        "total_samples": len(full_audio),
        "turns": timeline
    }

    return ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic multi-speaker test audio.")
    parser.add_argument("--output", default="samples/multi_speaker_test.wav", help="Output WAV path")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate in Hz")
    args = parser.parse_args()

    meta = generate_synthetic_dialogue(args.output, args.sr)
    print(f"Generated synthetic test file: {meta['output_path']}")
    print(f"Total Duration: {meta['total_duration']}s | Sample Rate: {meta['sample_rate']} Hz")
    for t in meta["turns"]:
        print(f"  [{t['start']}s -> {t['end']}s] {t['speaker']} (F0={t['f0']} Hz)")

