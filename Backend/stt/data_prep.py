"""
KisanCall STT — Data Preparation
Downloads Common Voice Punjabi/Hindi + generates synthetic agricultural phrases.
"""

import argparse
import os
import json
from pathlib import Path


# Agricultural phrases in Punjabi (Gurmukhi)
AGRI_PHRASES_PA = [
    "ਕਣਕ ਦੀ ਫ਼ਸਲ ਵਿੱਚ ਪੀਲਾਪਣ ਆ ਗਿਆ ਹੈ",
    "ਸਰ੍ਹੋਂ ਦੇ ਪੱਤੇ ਝੜ ਰਹੇ ਨੇ",
    "ਮੰਡੀ ਵਿੱਚ ਕਣਕ ਦਾ ਭਾਅ ਕੀ ਹੈ",
    "ਯੂਰੀਆ ਖਾਦ ਕਿੰਨੀ ਪਾਉਣੀ ਹੈ",
    "ਝੋਨੇ ਵਿੱਚ ਕੀੜੇ ਲੱਗ ਗਏ ਨੇ",
    "ਕਣਕ ਦੀ ਬਿਜਾਈ ਕਦੋਂ ਕਰਨੀ ਹੈ",
    "ਫ਼ਸਲ ਵਿੱਚ ਫੰਗਲ ਰੋਗ ਲੱਗ ਗਿਆ",
    "ਮਿੱਟੀ ਦੀ ਜਾਂਚ ਕਿਵੇਂ ਕਰਾਈਏ",
    "DAP ਖਾਦ ਦੀ ਕੀਮਤ ਕੀ ਹੈ",
    "ਝੋਨੇ ਦੀ ਫ਼ਸਲ ਨੂੰ ਪਾਣੀ ਕਦੋਂ ਲਾਉਣਾ ਹੈ",
    "ਕਣਕ ਦੀ ਕਟਾਈ ਦਾ ਸਮਾਂ ਕੀ ਹੈ",
    "ਪੱਤਿਆਂ ਤੇ ਭੂਰੇ ਧੱਬੇ ਆ ਗਏ ਨੇ",
    "PM ਕਿਸਾਨ ਯੋਜਨਾ ਲਈ ਅਪਲਾਈ ਕਿਵੇਂ ਕਰੀਏ",
    "ਅੱਜ ਮੌਸਮ ਕਿਹੋ ਜਿਹਾ ਰਹੇਗਾ",
]

# Agricultural phrases in Hindi (Devanagari)
AGRI_PHRASES_HI = [
    "गेहूं की फसल में पीलापन आ गया है",
    "सरसों के पत्ते झड़ रहे हैं",
    "मंडी में गेहूं का भाव क्या है",
    "यूरिया खाद कितनी डालनी है",
    "धान में कीड़े लग गए हैं",
    "गेहूं की बुवाई कब करनी है",
    "फसल में फफूंद रोग लग गया",
    "मिट्टी की जांच कैसे कराएं",
    "DAP खाद की कीमत क्या है",
    "धान की फसल को पानी कब देना है",
    "गेहूं की कटाई का समय क्या है",
    "पत्तों पर भूरे धब्बे आ गए हैं",
    "PM किसान योजना के लिए अप्लाई कैसे करें",
]

# Romanized agricultural phrases (common in STT output)
AGRI_PHRASES_ROMAN = [
    "kanak di fasal vich peelapa aa gaya hai",
    "sarson de patte jhar rahe ne",
    "mandi vich kanak da bhav ki hai",
    "urea khaad kinni pauni hai",
    "jhone vich keede lag gaye ne",
    "kanak di bijai kadon karni hai",
    "fasal vich fungal rog lag gaya",
    "mitti di jaanch kiven karaiye",
    "gehun ki fasal mein peelapan aa gaya hai",
    "mandi mein gehun ka bhav kya hai",
]


def generate_synthetic_audio(phrases, language, output_dir, repeats=3):
    """Generate synthetic audio using gTTS for agricultural phrases."""
    try:
        from gtts import gTTS
    except ImportError:
        print("WARNING: gTTS not installed. Skipping synthetic audio generation.")
        print("Install with: pip install gTTS")
        return []

    audio_dir = Path(output_dir) / "synthetic" / language
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    lang_code = "pa" if language == "punjabi" else "hi"

    for idx, phrase in enumerate(phrases):
        for rep in range(repeats):
            filename = f"agri_{language}_{idx:03d}_rep{rep}.mp3"
            filepath = audio_dir / filename

            if not filepath.exists():
                try:
                    tts = gTTS(text=phrase, lang=lang_code)
                    tts.save(str(filepath))
                    print(f"  Generated: {filename}")
                except Exception as e:
                    print(f"  ERROR generating {filename}: {e}")
                    continue

            manifest.append({
                "audio_path": str(filepath),
                "sentence": phrase,
                "language": lang_code,
                "source": "synthetic",
            })

    return manifest


def download_common_voice(language, max_samples, output_dir):
    """Download Common Voice dataset samples."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("WARNING: datasets not installed. Skipping Common Voice download.")
        return []

    lang_map = {"punjabi": "pa-IN", "hindi": "hi"}
    lang_code = lang_map.get(language, language)

    print(f"Downloading Common Voice {lang_code} (max {max_samples} samples)...")

    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            lang_code,
            split=f"train[:{max_samples}]",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"WARNING: Could not download Common Voice {lang_code}: {e}")
        print("Falling back to synthetic data only.")
        return []

    audio_dir = Path(output_dir) / "common_voice" / language
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, sample in enumerate(dataset):
        audio_path = audio_dir / f"cv_{language}_{idx:05d}.wav"

        # Save audio if not already saved
        if not audio_path.exists():
            try:
                import soundfile as sf
                audio_array = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                sf.write(str(audio_path), audio_array, sr)
            except Exception as e:
                print(f"  Error saving sample {idx}: {e}")
                continue

        manifest.append({
            "audio_path": str(audio_path),
            "sentence": sample["sentence"],
            "language": "pa" if language == "punjabi" else "hi",
            "source": "common_voice",
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{max_samples} samples")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="KisanCall STT Data Preparation")
    parser.add_argument("--output_dir", type=str, default="./stt/data",
                        help="Output directory for prepared data")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max Common Voice samples per language")
    parser.add_argument("--repeats", type=int, default=3,
                        help="How many times to repeat synthetic phrases")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip Common Voice download, use synthetic only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_manifest = []

    # Step 1: Generate synthetic agricultural audio
    print("\n=== Generating Synthetic Agricultural Phrases ===")

    print("\n--- Punjabi phrases ---")
    pa_synth = generate_synthetic_audio(
        AGRI_PHRASES_PA, "punjabi", output_dir, repeats=args.repeats
    )
    all_manifest.extend(pa_synth)
    print(f"Generated {len(pa_synth)} Punjabi synthetic samples")

    print("\n--- Hindi phrases ---")
    hi_synth = generate_synthetic_audio(
        AGRI_PHRASES_HI, "hindi", output_dir, repeats=args.repeats
    )
    all_manifest.extend(hi_synth)
    print(f"Generated {len(hi_synth)} Hindi synthetic samples")

    # Step 2: Download Common Voice (optional)
    if not args.skip_download:
        print("\n=== Downloading Common Voice Datasets ===")

        print("\n--- Common Voice Punjabi ---")
        pa_cv = download_common_voice("punjabi", args.max_samples, output_dir)
        all_manifest.extend(pa_cv)
        print(f"Downloaded {len(pa_cv)} Punjabi CV samples")

        print("\n--- Common Voice Hindi ---")
        hi_cv = download_common_voice("hindi", args.max_samples, output_dir)
        all_manifest.extend(hi_cv)
        print(f"Downloaded {len(hi_cv)} Hindi CV samples")

    # Step 3: Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Data Preparation Complete!")
    print(f"{'='*50}")
    print(f"Total samples: {len(all_manifest)}")
    print(f"  Synthetic Punjabi: {len(pa_synth)}")
    print(f"  Synthetic Hindi: {len(hi_synth)}")
    if not args.skip_download:
        print(f"  Common Voice Punjabi: {len(pa_cv)}")
        print(f"  Common Voice Hindi: {len(hi_cv)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
