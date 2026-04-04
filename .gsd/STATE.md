# STATE.md — Project Memory

## Current Position
- **Phase**: 1 — Backend Foundation & STT Module
- **Task**: Plan 1.1 partially complete (PyTorch downloading)
- **Status**: In progress — dependency installation running

## Active Background Process
- PyTorch CUDA 12.1 downloading (~2.4GB at ~468KB/s, ~1.5 hours remaining)
- Command ID: 5df60490-bfc0-4ff1-9cf9-cced3eab05ec

## Completed
- [x] Python 3.12 installed via winget
- [x] Virtual environment created (Backend/venv/)
- [x] Backend directory structure created (stt/, guardrails/, router/, response/, api/)
- [x] requirements.txt written (adjusted for 6GB VRAM)  
- [x] README.md written
- [x] stt/data_prep.py written (Common Voice + synthetic agri phrases)
- [x] stt/finetune.py written (LoRA, batch_size=1 for 6GB)
- [x] stt/inference.py written (STTEngine + Silero VAD)
- [x] .gitignore created

## Pending
- [ ] PyTorch + torchaudio install (downloading)
- [ ] Remaining pip dependencies (transformers, peft, oumi, etc.)
- [ ] Verify GPU access
- [ ] Run data_prep.py to generate synthetic agri audio
- [ ] Test STTEngine initialization

## Hardware Profile
- **GPU**: NVIDIA RTX 3050, 6GB VRAM
- **Python**: 3.12.10 (venv), 3.13.5 (system)
- **OS**: Windows
- **Constraints**: batch_size=1 required for fine-tuning, QLoRA needed for Phi-3

## Next Steps (after PyTorch finishes)
1. Install remaining dependencies: `.\Backend\venv\Scripts\pip.exe install -r Backend\requirements.txt`
2. Verify CUDA: `.\Backend\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`
3. Run data_prep: `.\Backend\venv\Scripts\python.exe -m stt.data_prep --output_dir ./stt/data --skip_download`
4. Continue to Plan 1.2 / 1.3 execution
