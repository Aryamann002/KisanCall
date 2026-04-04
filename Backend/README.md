# KisanCall Backend — Voice AI Pipeline

> 4 fine-tuned SLMs for agricultural voice advisory in Punjabi & Hindi

## Architecture

```
🎙️ Whisper STT (244M, LoRA) → 🛡️ Guardrails (Phi-3, 3.8B) → 🔀 Router (Phi-3, 3.8B) → 💬 Response (Phi-3, 3.8B) → 🔊 TTS
```

## Prerequisites

- Python 3.12 (3.13 has compatibility issues with whisper)
- NVIDIA GPU with CUDA (RTX 3050 6GB minimum)
- ~10GB disk for models

## Setup

```bash
cd Backend
py -3.12 -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Modules

| Module | Location | Model | Purpose |
|--------|----------|-------|---------|
| STT | `stt/` | Whisper Small (244M) | Speech-to-text for Punjabi/Hindi |
| Guardrails | `guardrails/` | Phi-3 mini (3.8B) | Farming/safety classifier |
| Router | `router/` | Phi-3 mini (3.8B) | Intent classification |
| Response | `response/` | Phi-3 mini (3.8B) | Agricultural advice generation |
| API | `api/` | N/A | FastAPI server |

## Training Pipeline

```bash
# 1. Prepare STT data
python -m stt.data_prep --output_dir ./stt/data

# 2. Fine-tune Whisper
python -m stt.finetune --data_dir ./stt/data --output_dir ./stt/model

# 3. Prepare Guardrails data
python -m guardrails.data_prep --output_dir ./guardrails/data

# 4. Fine-tune Guardrails (via Oumi)
oumi train -c guardrails/config/train.yaml

# 5. Run API server
python -m api.main
```

## GPU Notes (RTX 3050 6GB)

- Whisper Small: fits comfortably (244M params)
- Phi-3 mini: requires 4-bit quantization (QLoRA) for fine-tuning
- Inference: int4 quantization via bitsandbytes
- Batch size: 1 for training, gradient accumulation compensates
