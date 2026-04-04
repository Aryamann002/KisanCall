---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Backend Project Setup & Dependencies

## Objective
Set up the Python backend project structure with all ML dependencies installed. Create a virtual environment with Python 3.11/3.12 (required for PyTorch/Whisper compatibility — Python 3.13 has known issues with openai-whisper and ML libraries). Install PyTorch, transformers, faster-whisper, Oumi, and all required packages.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- .gsd/STACK.md

## Tasks

<task type="auto" effort="medium">
  <name>Create Backend project structure and virtual environment</name>
  <files>Backend/requirements.txt, Backend/README.md</files>
  <action>
    1. Check if Python 3.11 or 3.12 is available on the system (py -3.12 --version or py -3.11 --version). 
       - If not available, install Python 3.12 via winget or direct download
       - Python 3.13 has known incompatibilities with openai-whisper and some ML build systems
    
    2. Create a Python virtual environment in Backend/:
       ```
       cd Backend
       py -3.12 -m venv venv   (or python -m venv venv if 3.12 is default)
       ```
    
    3. Create the directory structure:
       ```
       Backend/
       ├── stt/              # Whisper STT module
       │   ├── __init__.py
       │   ├── data_prep.py
       │   ├── finetune.py
       │   └── inference.py
       ├── guardrails/       # Guardrails SLM module
       │   ├── __init__.py
       │   ├── data_prep.py
       │   ├── config/
       │   └── inference.py
       ├── router/           # Intent Router module
       │   ├── __init__.py
       │   └── inference.py
       ├── response/         # Response Generator module
       │   ├── __init__.py
       │   └── inference.py
       ├── api/              # FastAPI server
       │   ├── __init__.py
       │   └── main.py
       ├── requirements.txt
       ├── README.md
       └── venv/
       ```

    4. Create requirements.txt with pinned versions:
       ```
       torch>=2.2.0
       torchaudio>=2.2.0
       transformers>=4.40.0
       datasets>=2.19.0
       accelerate>=0.30.0
       peft>=0.11.0
       oumi
       faster-whisper>=1.0.0
       silero-vad>=5.0
       gTTS>=2.5.0
       fastapi>=0.111.0
       uvicorn>=0.29.0
       python-multipart>=0.0.9
       soundfile>=0.12.0
       librosa>=0.10.0
       numpy<2.0
       ```
    
    AVOID: Using Python 3.13 — it has broken metadata extraction for openai-whisper
    AVOID: Installing without CUDA — check for GPU first and install torch with CUDA support
  </action>
  <verify>
    Activate venv and run: python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    Should print torch version and True for CUDA.
  </verify>
  <done>Virtual environment created with all dependencies installed. PyTorch reports CUDA available.</done>
</task>

<task type="auto" effort="low">
  <name>Create Backend README with setup instructions</name>
  <files>Backend/README.md</files>
  <action>
    Create a README.md documenting:
    - Project overview (KisanCall Backend — 4 SLM cascade)
    - Prerequisites (Python 3.12, CUDA GPU)
    - Setup instructions (venv, pip install)
    - Module descriptions (stt, guardrails, router, response, api)
    - How to run each module
  </action>
  <verify>cat Backend/README.md — should contain all sections</verify>
  <done>README.md exists with complete setup documentation</done>
</task>

## Success Criteria
- [ ] Backend/ has the complete directory structure with __init__.py files
- [ ] Virtual environment created with Python 3.11 or 3.12
- [ ] All ML dependencies installed (torch, transformers, oumi, faster-whisper, etc.)
- [ ] PyTorch confirms CUDA GPU access
- [ ] README.md documents the setup process
