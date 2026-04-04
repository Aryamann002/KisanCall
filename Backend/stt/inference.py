"""
KisanCall STT — Inference Engine with Silero VAD
Real-time speech-to-text for Punjabi and Hindi agricultural queries.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class STTEngine:
    """
    Speech-to-Text engine using fine-tuned Whisper + Silero VAD.
    
    Pipeline:
      Audio → VAD (detect speech, ~3ms) → Whisper (transcribe, ~120ms)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_faster_whisper: bool = True,
    ):
        """
        Initialize STT engine.
        
        Args:
            model_path: Path to fine-tuned Whisper model. If None, uses base whisper-small.
            device: 'cuda', 'cpu', or 'auto'
            use_faster_whisper: Use faster-whisper for lower latency (CTranslate2 backend)
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"STTEngine initializing on {self.device}...")

        # Load Silero VAD
        self._load_vad()

        # Load Whisper
        self.use_faster_whisper = use_faster_whisper
        if use_faster_whisper:
            self._load_faster_whisper(model_path)
        else:
            self._load_whisper(model_path)

        print("STTEngine ready!")

    def _load_vad(self):
        """Load Silero Voice Activity Detector (1.8MB, ~3ms inference)."""
        print("  Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (
            self.get_speech_timestamps,
            _,
            self.read_audio,
            _,
            _,
        ) = self.vad_utils
        print("  VAD loaded (Silero, 1.8MB)")

    def _load_faster_whisper(self, model_path: Optional[str]):
        """Load faster-whisper for optimized inference."""
        print("  Loading faster-whisper...")
        try:
            from faster_whisper import WhisperModel

            model_size = model_path if model_path else "small"
            compute_type = "float16" if self.device == "cuda" else "int8"

            self.whisper = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
            )
            print(f"  Whisper loaded ({model_size}, {compute_type})")
        except ImportError:
            print("  WARNING: faster-whisper not available, falling back to transformers")
            self.use_faster_whisper = False
            self._load_whisper(model_path)

    def _load_whisper(self, model_path: Optional[str]):
        """Load Whisper via transformers (HuggingFace)."""
        print("  Loading Whisper via transformers...")
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = model_path if model_path else "openai/whisper-small"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.whisper_model.eval()
        print(f"  Whisper loaded ({model_id})")

    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect if audio contains speech using Silero VAD.
        
        Args:
            audio: Audio array (float32, mono)
            sample_rate: Sample rate (must be 16000)
            
        Returns:
            True if speech detected, False if silence/noise
        """
        start = time.perf_counter()

        audio_tensor = torch.from_numpy(audio).float()
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=sample_rate,
            threshold=0.5,
        )

        latency = (time.perf_counter() - start) * 1000
        has_speech = len(speech_timestamps) > 0

        return has_speech

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            audio_array: Audio as numpy array (16kHz, float32)
            language: Force language ('pa' for Punjabi, 'hi' for Hindi, or None for auto-detect)
            
        Returns:
            dict with 'text', 'language', 'latency_ms'
        """
        start = time.perf_counter()

        if self.use_faster_whisper:
            result = self._transcribe_faster_whisper(audio_path, audio_array, language)
        else:
            result = self._transcribe_transformers(audio_path, audio_array, language)

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    def _transcribe_faster_whisper(
        self, audio_path, audio_array, language
    ) -> dict:
        """Transcribe using faster-whisper."""
        if audio_array is not None:
            # faster-whisper can accept numpy arrays directly
            segments, info = self.whisper.transcribe(
                audio_array,
                language=language,
                beam_size=1,  # Greedy for speed
                vad_filter=True,
            )
        else:
            segments, info = self.whisper.transcribe(
                audio_path,
                language=language,
                beam_size=1,
                vad_filter=True,
            )

        text = " ".join([seg.text for seg in segments]).strip()

        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }

    def _transcribe_transformers(
        self, audio_path, audio_array, language
    ) -> dict:
        """Transcribe using HuggingFace transformers."""
        if audio_array is None:
            import librosa
            audio_array, _ = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)

        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return {
            "text": text,
            "language": language or "auto",
        }


# ── CLI for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KisanCall STT Inference")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to fine-tuned model")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file to transcribe")
    parser.add_argument("--language", type=str, default=None,
                        help="Force language (pa/hi)")
    parser.add_argument("--no-faster", action="store_true",
                        help="Don't use faster-whisper")
    args = parser.parse_args()

    engine = STTEngine(
        model_path=args.model,
        use_faster_whisper=not args.no_faster,
    )

    if args.audio:
        result = engine.transcribe(audio_path=args.audio, language=args.language)
        print(f"\nTranscription: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Latency: {result['latency_ms']}ms")
    else:
        print("\nSTTEngine initialized successfully!")
        print("Use --audio <path> to transcribe a file.")
        print(f"Device: {engine.device}")
