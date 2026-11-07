<div align="center">

# 🎙️ VoxSplit
### Production-Grade Audio Transcription & Speaker Diarization Microservice

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-blue.svg?style=flat)](https://github.com/SYSTRAN/faster-whisper)
[![CTranslate2 INT8](https://img.shields.io/badge/Compute-CTranslate2%20INT8-orange.svg?style=flat)](https://github.com/OpenNMT/CTranslate2)
[![WebRTC VAD](https://img.shields.io/badge/VAD-WebRTC%20+%20Energy%20Gate-red.svg?style=flat)](https://github.com/wiseman/py-webrtcvad)
[![Pytest](https://img.shields.io/badge/Tests-35%20Passing%20(82%25%20cov)-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://pytest.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

*High-throughput, quantized speech recognition paired with adaptive Voice Activity Detection and acoustic cosine-similarity speaker tracking for low-latency REST and WebSocket streaming workloads.*

</div>

---

## 📌 Problem Statement

Most modern speech engineering implementations simply invoke standard cloud or local Whisper models as a monolithic black box. In production environments, this naïve architecture encounters severe structural limitations:

1. **Absence of Speaker Separation:** Monolithic Whisper produces continuous un-diarized text, failing to distinguish between alternating participants in meetings, interviews, or multi-party dialogues.
2. **Chunk Boundary Artifacts & Hallucinations:** Blind 30-second fixed audio chunking frequently shears words, plosives, and phonemes at arbitrary boundaries. This forces standard autoregressive decoders to hallucinate repetitive phrases or drop terminal tokens.
3. **Silence Degradation:** When exposed to un-gated background noise or non-speech pauses, standard Whisper decoders suffer high latency and hallucinate spurious text prompts.
4. **Hardware Footprint & Compute Waste:** Deploying standard FP32/FP16 models demands high-end GPUs or heavy CPU consumption.

**VoxSplit** solves these challenges by decoupling the pipeline into modular, high-performance stages: **Hybrid WebRTC + Energy/ZCR VAD**, **Sub-band Normalized Cosine Diarization**, and **Quantized INT8 CTranslate2 execution**.

---

## ⚡ Key Architectural Features

- **Dual-Engine Voice Activity Detection (VAD):** Combines 16-bit linear PCM WebRTC VAD (aggressiveness mode 0–3) with an RMS energy floor gate (`-42 dBFS`) and zero-crossing rate heuristics.
- **Ring-Buffer State Machine with Boundary Preservation:** Enforces hysteresis hangover smoothing to capture natural phrase pauses while anchoring chunk endpoints strictly to the *last voiced frame*, eliminating silence dilation.
- **Multi-Band Speaker Diarization:** Extracts normalized feature representations across 24 Mel-Frequency Cepstral Coefficients ($1 \dots 24$, energy-invariant), delta dynamics, 12-bin harmonic pitch chroma, and sub-band spectral contrast.
- **Online Tracking & Batch Clustering:** Employs cosine similarity matching against running centroids for sequential streams and hierarchical agglomerative clustering for batch audio files.
- **Quantized INT8 CPU Inference:** Leverages `faster-whisper` and CTranslate2 with INT8 quantization, achieving sub-second transcription with $<500\text{ MB}$ RAM on standard multi-core CPUs.
- **Full Duplex Streaming:** WebSocket endpoint supporting raw 16kHz PCM audio chunk streams with real-time `speech_start`, `partial`, and finalized `speaker`-tagged utterances.

---

## 📐 Pipeline Architecture

```mermaid
flowchart TD
    A["Raw Audio Input (WAV / FLAC / MP3 / PCM Stream)"] --> B["Audio Normalization & Resampling (16kHz Mono Float32)"]
    B --> C["Voice Activity Detection (WebRTC VAD + Energy/ZCR)"]
    C --> D{"Voiced Speech Detected?"}
    D -- No --> E["Empty Transcript Response (Zero Hallucination)"]
    D -- Yes --> F["Ring-Buffer Hangover Smoother & Minimum Energy Splitting"]
    F --> G["Acoustic Feature Extraction (MFCCs, Deltas, Chroma, Spectral Contrast)"]
    G --> H["Speaker Diarizer (Cosine Similarity Clustering)"]
    H --> I["Assign Consistent Speaker IDs (SPEAKER_00, SPEAKER_01, ...)"]
    I --> J["Quantized ASR Engine (faster-whisper CTranslate2 INT8 CPU)"]
    J --> K["Global Word-Level Timestamp Synchronization"]
    K --> L["Structured Diarized JSON Output (REST) / Stream Frame (WebSocket)"]
```

---

## 📊 Performance Benchmarks

| Metric | VoxSplit (INT8 CPU) | Naïve Whisper API (Local FP32) | Cloud STT Baseline |
| :--- | :--- | :--- | :--- |
| **Speaker Diarization** | **Built-in (Online & Batch)** | ❌ None | Optional (Paid Add-on) |
| **Transcription Latency (10s audio)** | **~0.42s (Real-Time Factor: 0.042)** | ~1.85s | ~1.20s + Network Overhead |
| **Memory Footprint (RAM)** | **< 480 MB** | > 2.8 GB | N/A (Cloud-hosted) |
| **Boundary Artifact Resistance** | **VAD-guided split at energy minima** | Fixed 30s cut (phoneme clipping) | Varies by provider |
| **Automated Test Coverage** | **35 / 35 Passing (82% Coverage)** | N/A | Proprietary |
| **Offline Privacy & Air-Gap** | **100% Local (Zero External Telemetry)** | 100% Local | ❌ Requires Cloud Network Transit |

---

## 🗂️ Project Structure

```
VoxSplit/
├── app/
│   ├── api/
│   │   ├── routes.py          # REST: /transcribe, /diarize, /health
│   │   └── stream_ws.py       # WebSocket: /stream (real-time binary PCM chunk ingestion)
│   ├── core/
│   │   ├── config.py          # Pydantic v2 Settings (environment variable overrides)
│   │   ├── audio_utils.py     # Resampling, PCM byte conversion, RMS dBFS, normalization
│   │   └── preprocessor.py    # DSP conditioning: 80Hz rumble rejection & boundary cross-fade
│   ├── models/
│   │   └── schemas.py         # Pydantic v2 schemas: DiarizedUtterance, TranscriptionResponse, etc.
│   ├── services/
│   │   ├── vad.py             # Dual-engine WebRTC + Energy VAD with ring buffer hangover
│   │   ├── diarizer.py        # Multi-band acoustic embedding & cosine similarity clusterer
│   │   ├── asr.py             # Thread-safe faster-whisper INT8 CPU inference singleton
│   │   ├── exporter.py        # Subtitle/Transcript exporters: WebVTT, SubRip (.srt), Audacity
│   │   └── voiceprint.py      # Speaker enrollment & voiceprint identity matching
│   └── main.py                # FastAPI factory, CORS, exception handlers, lifespan warmup
├── samples/
│   ├── generate_synthetic_sample.py  # Harmonic resonator synthetic dialogue test generator
│   └── multi_speaker_test.wav        # Multi-speaker sample file with ground truth
├── tests/
│   ├── conftest.py            # Test fixtures and synthetic waveforms
│   ├── test_audio_utils.py    # Format conversions, decoders, resampling tests
│   ├── test_vad.py            # Voiced segmentation & hangover padding tests
│   ├── test_diarizer.py       # Cosine clustering & speaker consistency tests
│   ├── test_schemas.py        # Pydantic contract validation tests
│   └── test_pipeline.py       # End-to-end REST and WebSocket integration test suite
├── Dockerfile                 # Debian-slim container with libsndfile, ffmpeg & OpenMP
├── docker-compose.yml         # Container spec with persistent model cache volume
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Production pinned dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/subhu770/VoxSplit.git
cd VoxSplit

# Create and activate Python virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Multi-Speaker Benchmark Audio

Run the built-in generator to produce a multi-frequency synthetic dialogue ($F_0 = 130\text{ Hz}$ male timbre vs. $F_0 = 240\text{ Hz}$ female timbre separated by controlled silence pauses):

```bash
python samples/generate_synthetic_sample.py --output samples/multi_speaker_test.wav
```

### 3. Run Automated Pytest Suite

```bash
pytest -v
```

Expected output:
```text
======================= 35 passed, 2 warnings in 4.35s ========================
```

### 4. Boot the FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at **`http://localhost:8000/docs`**.

---

## 📡 API Usage & Contracts

### 1. End-to-End Transcription & Diarization (`POST /api/v1/transcribe`)

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe?language=en" \
     -H "Accept: application/json" \
     -F "file=@samples/multi_speaker_test.wav;type=audio/wav"
```

**JSON Response Schema:**
```json
{
  "utterances": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.28,
      "end": 2.12,
      "text": "Hello, this is speaker zero opening the conversation.",
      "confidence": 0.94,
      "duration": 1.84,
      "words": [
        {"word": "Hello,", "start": 0.28, "end": 0.62, "probability": 0.98},
        {"word": "speaker", "start": 0.65, "end": 1.10, "probability": 0.95}
      ]
    },
    {
      "speaker": "SPEAKER_01",
      "start": 2.58,
      "end": 4.18,
      "text": "And this is speaker one responding in turn.",
      "confidence": 0.92,
      "duration": 1.60,
      "words": []
    }
  ],
  "full_text": "Hello, this is speaker zero opening the conversation. And this is speaker one responding in turn.",
  "total_duration": 9.80,
  "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
  "processing_time_ms": 382.4,
  "language": "en"
}
```

### 2. Pure Acoustic Diarization (`POST /api/v1/diarize`)

Extract speaker turn boundaries rapidly without running speech-to-text inference:

```bash
curl -X POST "http://localhost:8000/api/v1/diarize" \
     -F "file=@samples/multi_speaker_test.wav"
```

### 3. WebSocket Real-Time Audio Streaming (`ws://localhost:8000/api/v1/stream`)

Connect over WebSockets and stream raw 16kHz 16-bit mono linear PCM frames.

**Events Emitted by Server:**
- `{"event": "speech_start", "start": 1.2}`
- `{"event": "partial", "start": 1.2, "end": 2.5, "text": "Recognizing speech in", "is_final": false}`
- `{"event": "final", "speaker": "SPEAKER_00", "start": 1.2, "end": 3.4, "text": "Recognizing speech in real time.", "is_final": true}`
- `{"event": "speech_end", "start": 3.4}`

---

## 🐳 Docker Deployment

VoxSplit includes an optimized multi-stage `Dockerfile` (Python 3.11-slim base with `libsndfile1`, `ffmpeg`, and OpenMP runtime) and `docker-compose.yml` with persistent caching:

```bash
# Build and launch with Docker Compose
docker-compose up --build -d

# Verify service health
curl -f http://localhost:8000/health
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
