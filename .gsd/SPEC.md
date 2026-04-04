# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
KisanCall is a Voice AI platform for Indian farmers that runs entirely offline on a local computer. Farmers speak in Punjabi or Hindi and receive instant expert advice on crop diseases, mandi prices, weather, and government schemes — all powered by a cascade of 4 fine-tuned Small Language Models (SLMs) under 10B parameters each. Built for the Oumi Hackathon track at Eclipse 6.0 (Thapar University, Team EL BICHO, PS ID EC602).

## Goals
1. **Build a working offline ML backend** — Whisper STT, Guardrails SLM, Intent Router SLM, Response Generator SLM, and TTS all running locally on the user's computer
2. **Fine-tune models using Oumi** — Meet hackathon requirement of using Oumi for LoRA fine-tuning of Phi-3 mini models
3. **Achieve <500ms end-to-end latency** — From voice input to voice response on consumer hardware
4. **Connect frontend demo to real backend** — Replace Anthropic Claude API calls with local model inference
5. **Support Punjabi and Hindi** — Full pipeline in both languages including STT, guardrails, routing, response, and TTS

## Non-Goals (Out of Scope)
- IVR/toll-free number integration (Phase 3 roadmap item — not for hackathon)
- WhatsApp Business API integration
- Image/photo input for crop disease detection
- More than 2 languages (Punjabi + Hindi only)
- Cloud deployment or scaling to 50,000+ concurrent calls
- Mobile app development
- Production-grade infrastructure (Docker, Kubernetes, etc.)

## Users
**Primary:** Hackathon judges evaluating the demo for creativity, real-world impact, and technical quality.
**Target end-user (future):** Indian farmers in Punjab who speak Punjabi or Hindi and need agricultural advisory via voice.

## Constraints
- Must run offline on user's local Windows computer
- Must use **Oumi** framework for fine-tuning (hackathon requirement)
- Models must be SLMs (<10B params) — Whisper Small (244M), Phi-3 mini (3.8B)
- Must demonstrate real latency advantage over cloud LLMs
- GPU: Consumer-grade (16GB VRAM target — T4 equivalent or RTX 3080/4070 class)
- Time: Hackathon timeline (limited days)
- Open-weight models highly encouraged

## Success Criteria
- [ ] Whisper STT fine-tuned on Punjabi+Hindi with agricultural vocabulary, WER ≤16% on Punjabi
- [ ] Guardrails SLM classifies farming/non-farming queries with >90% accuracy
- [ ] Intent Router routes to correct category (crop_disease, mandi_price, weather, govt_scheme)
- [ ] Response Generator produces concise agricultural advice in the user's language
- [ ] Full pipeline runs offline with <500ms end-to-end latency
- [ ] Frontend connects to local backend API instead of Anthropic Claude
- [ ] TTS speaks responses back in Punjabi/Hindi

## Technical Architecture (4 SLM Cascade)

```
🎙️ Whisper STT (244M, LoRA fine-tuned)
    → 🛡️ Guardrails (Phi-3 mini 3.8B, LoRA via Oumi)
        → 🔀 Intent Router (Phi-3 mini 3.8B, LoRA via Oumi)
            → 💬 Response Generator (Phi-3 mini 3.8B, LoRA via Oumi)
                → 🔊 TTS (gTTS/Kokoro/Piper)
```

## Key Components

| Module | Model | Params | Training Method | Latency Target |
|--------|-------|--------|-----------------|----------------|
| STT | Whisper Small | 244M | LoRA (8.4M trainable) | ~120ms |
| Guardrails | Phi-3 mini | 3.8B | LoRA via Oumi (rank 16) | ~68ms |
| Intent Router | Phi-3 mini | 3.8B | LoRA via Oumi | ~55ms |
| Response Generator | Phi-3 mini | 3.8B | LoRA via Oumi | ~180ms |
| VAD | Silero | 1.8MB | Pre-trained | ~3ms |
| TTS | gTTS/Piper | N/A | Pre-trained | ~55ms |
