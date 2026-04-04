# REQUIREMENTS.md

## Format
| ID | Requirement | Source | Status |
|----|-------------|--------|--------|
| REQ-01 | Whisper Small fine-tuned on Punjabi+Hindi with agri vocabulary, WER ≤16% Punjabi | SPEC goal 1 | Pending |
| REQ-02 | Guardrails SLM classifies farming/non-farming with >90% accuracy, <50ms | SPEC goal 2 | Pending |
| REQ-03 | Intent Router routes queries to 4 categories (crop_disease, mandi_price, weather, govt_scheme) | SPEC goal 2 | Pending |
| REQ-04 | Response Generator produces concise agricultural advice in user's language | SPEC goal 2 | Pending |
| REQ-05 | Full pipeline runs offline on local Windows machine with consumer GPU | SPEC goal 1 | Pending |
| REQ-06 | Models fine-tuned using Oumi framework (hackathon requirement) | SPEC constraint | Pending |
| REQ-07 | Frontend demo connects to local backend API (replaces Anthropic Claude calls) | SPEC goal 4 | Pending |
| REQ-08 | End-to-end latency <500ms from voice input to voice response | SPEC goal 3 | Pending |
| REQ-09 | TTS speaks responses in Punjabi/Hindi | SPEC goal 5 | Pending |
