# STATE.md — Project Memory

## Last Session Summary

Codebase mapping complete (2026-04-05).

- 1 main component identified (Frontend landing page — monolithic HTML)
- 0 production dependencies (no package manager)
- 2 external APIs active (Anthropic Claude, Web Speech API)
- 6 planned ML models (Whisper, 3× Phi-3, Silero VAD, gTTS)
- 11 technical debt items found
- Backend directory empty — no server implementation

## Current Position

- **Phase:** Pre-initialization (mapping complete, /new-project pending)
- **Last Action:** /map — codebase analyzed
- **Next Steps:** Run /new-project to define SPEC.md and project roadmap

## Key Findings

1. Frontend is a 2,347-line monolithic HTML file with everything inline
2. ML pipeline is described but not implemented — demo uses Claude API as proxy
3. Anthropic API key potentially exposed in client-side JS
4. No build system, no tests, no backend
5. Design is high-quality dark theme with glassmorphism — ready for production polish
