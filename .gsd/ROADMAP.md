# ROADMAP.md

> **Current Phase**: Not started
> **Milestone**: v1.0 — Hackathon Demo (Offline Pipeline)

## Must-Haves (from SPEC)
- [ ] Whisper STT fine-tuned for Punjabi+Hindi agri vocab
- [ ] Guardrails SLM via Oumi (farming/safety classifier)
- [ ] Intent Router SLM via Oumi (4 categories)
- [ ] Response Generator SLM via Oumi (agricultural advice)
- [ ] TTS output in Punjabi/Hindi
- [ ] Backend API server connecting all models
- [ ] Frontend connected to local backend (not Anthropic API)
- [ ] Full pipeline running offline <500ms

## Phases

### Phase 1: Backend Foundation & STT Module
**Status**: 🔄 Code Complete (Waiting on PyTorch download)
**Objective**: Set up the Python backend project structure, install dependencies, and build the STT module with Whisper fine-tuning pipeline (data prep → LoRA training → inference engine with Silero VAD).
**Requirements**: REQ-01 (STT), REQ-05 (offline)

### Phase 2: Guardrails SLM Module
**Status**: 🔄 Code Complete (Waiting on PyTorch download)
**Objective**: Build the Guardrails module — data prep with farming/non-farming examples, Oumi fine-tuning config for Phi-3 mini, and inference engine that classifies queries as ALLOW/REDIRECT/BLOCK.
**Requirements**: REQ-02 (Guardrails), REQ-06 (Oumi)

### Phase 3: Intent Router & Response Generator
**Status**: 🔄 Code Complete (Waiting on PyTorch download)
**Objective**: Build the Intent Router (route to crop_disease/mandi_price/weather/govt_scheme) and Response Generator (agricultural advice in user's language), both fine-tuned via Oumi.
**Requirements**: REQ-03 (Router), REQ-04 (Response)

### Phase 4: Pipeline Integration & Frontend Connection
**Status**: 🔄 Code Complete (Waiting on PyTorch download)
**Objective**: Wire all modules into a FastAPI backend, add TTS output, connect the frontend to the local API, and demonstrate the full offline pipeline with <500ms latency.
**Requirements**: REQ-05 (offline), REQ-07 (frontend), REQ-08 (latency)
