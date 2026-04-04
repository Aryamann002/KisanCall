"""
KisanCall STT — LoRA Fine-tuning for Whisper Small
Fine-tunes on Punjabi/Hindi agricultural audio. Adjusted for RTX 3050 6GB VRAM.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AgriAudioDataset(Dataset):
    """Custom dataset for agricultural audio samples."""

    def __init__(self, manifest_path, processor, max_samples=None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        sentence = sample["sentence"]

        # Load audio
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            # Return a short silence if audio can't be loaded
            audio = np.zeros(16000, dtype=np.float32)
            print(f"Warning: Could not load {audio_path}: {e}")

        # Process for Whisper
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        # Tokenize target text
        labels = self.processor.tokenizer(
            sentence, return_tensors="pt"
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


def collate_fn(batch):
    """Custom collation for variable-length labels."""
    input_features = torch.stack([item["input_features"] for item in batch])

    # Pad labels to same length
    label_lengths = [len(item["labels"]) for item in batch]
    max_label_len = max(label_lengths)

    labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        labels[i, :len(item["labels"])] = item["labels"]

    return {
        "input_features": input_features,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="KisanCall STT Fine-tuning")
    parser.add_argument("--data_dir", type=str, default="./stt/data",
                        help="Directory with manifest.json")
    parser.add_argument("--output_dir", type=str, default="./stt/model",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 for 6GB VRAM)")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (for quick testing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load Whisper Small
    print("\nLoading Whisper Small model...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float16,
    )
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Apply LoRA
    print("Applying LoRA adapters...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = model.to(device)

    # Load dataset
    manifest_path = Path(args.data_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run data_prep.py first!")
        return

    print(f"\nLoading dataset from {manifest_path}...")
    dataset = AgriAudioDataset(manifest_path, processor, max_samples=args.max_samples)
    print(f"Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
    )

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_r}")

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            input_features = batch["input_features"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_features=input_features,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation

            loss.backward()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.gradient_accumulation
            num_batches += 1

            if (step + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step+1} | Loss: {avg_loss:.4f}")

            # Clear CUDA cache periodically (6GB VRAM management)
            if (step + 1) % 50 == 0:
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")

    # Save model
    print(f"\nSaving fine-tuned model to {args.output_dir}...")
    output_path = Path(args.output_dir) / "whisper-small-kisancall"
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge LoRA and save
    print("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    print(f"\n{'='*50}")
    print(f"Fine-tuning Complete!")
    print(f"{'='*50}")
    print(f"Model saved to: {output_path}")
    print(f"Total training steps: {global_step}")
    print(f"Final loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    main()
