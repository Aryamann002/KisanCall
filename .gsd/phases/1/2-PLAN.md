---
phase: 1
plan: 2
wave: 2
---

# Plan 1.2: STT Data Preparation

## Objective
Build the dataset for fine-tuning Whisper Small on Punjabi and Hindi agricultural audio. Download Common Voice datasets, generate 27+ synthetic agricultural phrases via gTTS, and prepare the training data in the format Whisper expects.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- Backend/requirements.txt
- Claude (1).pdf — STT module architecture details

## Tasks

<task type="auto" effort="high">
  <name>Create STT data preparation script</name>
  <files>Backend/stt/data_prep.py</files>
  <action>
    Create Backend/stt/data_prep.py that:

    1. Downloads Mozilla Common Voice Punjabi (pa-IN) and Hindi (hi) datasets from HuggingFace:
       ```python
       from datasets import load_dataset
       cv_pa = load_dataset("mozilla-foundation/common_voice_17_0", "pa-IN", split="train", trust_remote_code=True)
       cv_hi = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train", trust_remote_code=True)
       ```
       - Use streaming=True if dataset is too large
       - Take first 500-1000 samples per language for hackathon scope

    2. Generates 27+ synthetic agricultural phrases using gTTS in both Punjabi and Hindi:
       Punjabi phrases (examples):
       - "ਕਣਕ ਦੀ ਫ਼ਸਲ ਵਿੱਚ ਪੀਲਾਪਣ ਆ ਗਿਆ ਹੈ"
       - "ਸਰ੍ਹੋਂ ਦੇ ਪੱਤੇ ਝੜ ਰਹੇ ਨੇ"
       - "ਮੰਡੀ ਵਿੱਚ ਕਣਕ ਦਾ ਭਾਅ ਕੀ ਹੈ"
       - "ਯੂਰੀਆ ਖਾਦ ਕਿੰਨੀ ਪਾਉਣੀ ਹੈ"
       - "ਝੋਨੇ ਵਿੱਚ ਕੀੜੇ ਲੱਗ ਗਏ ਨੇ"
       Hindi phrases (examples):
       - "गेहूं की फसल में पीलापन आ गया है"
       - "मंडी में गेहूं का भाव क्या है"
       - "यूरिया खाद कितनी डालनी है"
       
       Generate audio via gTTS, save as WAV at 16kHz:
       ```python
       from gtts import gTTS
       import soundfile as sf
       ```
       
       Repeat each synthetic phrase 3× to bias Whisper toward farming vocabulary.

    3. Formats all data into a HuggingFace Dataset with columns:
       - audio (path or array)
       - sentence (ground truth text)
       - language (pa/hi)
       
    4. Saves the prepared dataset to Backend/stt/data/

    AVOID: Downloading the entire Common Voice dataset — use streaming and limit samples
    AVOID: Hardcoding absolute paths — use relative paths or argparse
    
    Include argparse with --output_dir flag.
  </action>
  <verify>
    cd Backend && python -m stt.data_prep --output_dir ./stt/data --max_samples 100
    Should create data directory with audio files and a dataset manifest.
  </verify>
  <done>
    - data_prep.py creates a combined dataset of Common Voice + synthetic agri phrases
    - At least 27 unique agricultural phrases generated in both languages
    - Each synthetic phrase repeated 3× in the dataset
    - Dataset saved in HuggingFace format at Backend/stt/data/
  </done>
</task>

## Success Criteria
- [ ] data_prep.py runs without errors
- [ ] Synthetic agricultural phrases generated as audio files
- [ ] Combined dataset contains both Common Voice and synthetic data
- [ ] Dataset is in the correct format for Whisper fine-tuning
