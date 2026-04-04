---
phase: 1
plan: 3
wave: 2
---

# Plan 1.3: STT Fine-Tuning & Inference Engine

## Objective
Create the LoRA fine-tuning script for Whisper Small on the prepared agricultural dataset, and build the real-time inference engine with Silero VAD for voice activity detection. This delivers a working STT module that transcribes Punjabi/Hindi farmer speech with <120ms latency.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- Backend/stt/data_prep.py (created in Plan 1.2)
- Claude (1).pdf — STT fine-tuning details

## Tasks

<task type="auto" effort="high">
  <name>Create Whisper LoRA fine-tuning script</name>
  <files>Backend/stt/finetune.py</files>
  <action>
    Create Backend/stt/finetune.py that:

    1. Loads the prepared dataset from Backend/stt/data/
    
    2. Loads Whisper Small (openai/whisper-small) with LoRA adapters:
       ```python
       from transformers import WhisperForConditionalGeneration, WhisperProcessor
       from peft import LoraConfig, get_peft_model
       
       model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
       processor = WhisperProcessor.from_pretrained("openai/whisper-small")
       
       lora_config = LoraConfig(
           r=16,
           lora_alpha=32,
           target_modules=["q_proj", "v_proj"],
           lora_dropout=0.05,
           bias="none",
       )
       model = get_peft_model(model, lora_config)
       ```
       
       This yields ~8.4M trainable params on top of 244M base.

    3. Sets up training with HuggingFace Trainer:
       - Batch size: 4 (fits on 16GB VRAM)
       - Learning rate: 1e-4
       - Epochs: 3
       - FP16 training
       - Gradient accumulation steps: 4
       - Save checkpoints every 500 steps
    
    4. After training, merges LoRA adapters into base weights:
       ```python
       model = model.merge_and_unload()
       model.save_pretrained("Backend/stt/model/whisper-small-kisancall")
       processor.save_pretrained("Backend/stt/model/whisper-small-kisancall")
       ```
    
    5. Include argparse with --data_dir, --output_dir, --epochs, --batch_size flags.

    AVOID: Training without FP16 — will OOM on consumer GPUs
    AVOID: Full fine-tuning — must use LoRA for memory efficiency
  </action>
  <verify>
    python -m stt.finetune --data_dir ./stt/data --output_dir ./stt/model --epochs 1 --batch_size 2
    Should start training and complete without OOM errors.
  </verify>
  <done>
    - finetune.py trains Whisper with LoRA (8.4M params)
    - Merged model saved to Backend/stt/model/
    - Training completes without OOM on 16GB GPU
  </done>
</task>

<task type="auto" effort="high">
  <name>Create STT inference engine with Silero VAD</name>
  <files>Backend/stt/inference.py</files>
  <action>
    Create Backend/stt/inference.py with an STTEngine class:

    1. STTEngine.__init__():
       - Load fine-tuned Whisper model (or base whisper-small as fallback)
       - Load Silero VAD model (1.8MB, ~3ms inference)
       - Initialize processor and feature extractor
       ```python
       import torch
       model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
       (get_speech_timestamps, _, read_audio, _, _) = utils
       ```
    
    2. STTEngine.detect_speech(audio_chunk) -> bool:
       - Run Silero VAD on audio chunk
       - Return True if speech detected, False if silence
       - ~3ms latency
    
    3. STTEngine.transcribe(audio_path_or_array) -> dict:
       - Load audio, resample to 16kHz
       - Run Whisper inference
       - Return {"text": transcription, "language": detected_lang, "latency_ms": time}
       - Target: ~120ms on GPU
    
    4. STTEngine.transcribe_stream(audio_stream) -> AsyncGenerator:
       - Continuous mic capture mode
       - VAD detects speech onset/offset
       - Transcribes only voiced segments
       - Yields transcription results
    
    Include a __main__ block for testing:
    ```python
    if __name__ == "__main__":
        engine = STTEngine()
        result = engine.transcribe("test_audio.wav")
        print(result)
    ```

    AVOID: Loading model on CPU — must use CUDA for latency targets
    AVOID: Processing silence through Whisper — always VAD-gate first
  </action>
  <verify>
    python -c "from stt.inference import STTEngine; e = STTEngine(); print('STTEngine loaded OK')"
    Should load without errors and confirm GPU usage.
  </verify>
  <done>
    - STTEngine class loads Whisper + Silero VAD
    - detect_speech() returns bool in <5ms
    - transcribe() processes audio in ~120ms on GPU
    - transcribe_stream() available for continuous input
  </done>
</task>

## Success Criteria
- [ ] Fine-tuning script trains without OOM on consumer GPU
- [ ] Merged model saved to disk
- [ ] STTEngine loads and runs inference
- [ ] Silero VAD correctly filters silence
- [ ] End-to-end STT latency <150ms on GPU
