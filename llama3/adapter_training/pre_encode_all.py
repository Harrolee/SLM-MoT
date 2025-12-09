#!/usr/bin/env python
"""
Pre-encode all MusicCaps audio files to tokens for training.
INTERRUPTABLE: Use Ctrl+C to pause, run again to resume.
"""

import torch
import numpy as np
from pathlib import Path
import json
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio
import signal
import sys
import time
from datetime import datetime, timedelta

# Global flag for graceful shutdown
interrupted = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global interrupted
    interrupted = True
    print("\n\n⚠️  INTERRUPT RECEIVED - Saving progress...")
    print("Run this script again to resume from where you left off.")

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def load_progress():
    """Load encoding progress from checkpoint file."""
    progress_file = Path("pre_encoding_progress.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        "completed": [],
        "failed": [],
        "total_processed": 0,
        "total_time_seconds": 0,
        "last_run": None
    }

def save_progress(progress):
    """Save encoding progress to checkpoint file."""
    progress_file = Path("pre_encoding_progress.json")
    progress["last_run"] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    print(f"✅ Progress saved to {progress_file}")

def encode_audio_file(audio_path, processor, musicgen, device="mps"):
    """Encode a single audio file to tokens."""
    # Force soundfile backend to avoid torchcodec issue
    import soundfile as sf

    # Load audio using soundfile directly
    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32).t()  # Transpose to [channels, samples]

    # Ensure shape is correct [channels, samples]
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Process with MusicGen
    inputs = processor(
        audio=waveform.squeeze().numpy(),
        sampling_rate=32000,
        return_tensors="pt"
    ).to(device)

    # Encode to tokens
    with torch.no_grad():
        audio_codes = musicgen.audio_encoder.encode(inputs["input_values"])[0]  # [batch, n_codebooks, seq_len]

    return audio_codes.cpu().numpy()

def main():
    print("=" * 60)
    print("PRE-ENCODING ALL MUSICCAPS AUDIO FILES")
    print("=" * 60)
    print("Press Ctrl+C at any time to pause safely")
    print("Run this script again to resume\n")

    # Setup paths
    audio_dir = Path("musiccaps_audio")
    output_dir = Path("pre_encoded_audio")
    output_dir.mkdir(exist_ok=True)

    # Load progress
    progress = load_progress()

    # Get all audio files
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    total_files = len(audio_files)

    # Filter out already completed files
    completed_set = set(progress["completed"])
    remaining_files = [f for f in audio_files if f.stem not in completed_set]

    print(f"📊 Status:")
    print(f"   Total files: {total_files}")
    print(f"   Already encoded: {len(completed_set)}")
    print(f"   Remaining: {len(remaining_files)}")
    print(f"   Failed (will retry): {len(progress['failed'])}")

    if len(remaining_files) == 0:
        print("\n✅ All files already encoded!")
        return

    # Load MusicGen model
    print("\n📦 Loading MusicGen model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
    print(f"   Device: {device}")

    # Start encoding
    print(f"\n🚀 Starting encoding...")
    start_time = time.time()
    session_encoded = 0
    errors = []

    for i, audio_file in enumerate(remaining_files):
        if interrupted:
            break

        ytid = audio_file.stem
        output_file = output_dir / f"{ytid}.npy"

        # Progress display
        total_progress = len(completed_set) + i
        percent = (total_progress / total_files) * 100

        # Time estimation
        if session_encoded > 0:
            time_per_file = (time.time() - start_time) / session_encoded
            remaining_time = time_per_file * (len(remaining_files) - i)
            eta = timedelta(seconds=int(remaining_time))
            eta_str = f"ETA: {eta}"
        else:
            eta_str = "Calculating..."

        print(f"\n[{total_progress+1}/{total_files}] ({percent:.1f}%) - {eta_str}")
        print(f"   Encoding: {ytid}")

        try:
            # Encode audio
            audio_codes = encode_audio_file(audio_file, processor, musicgen, device)

            # Save encoded tokens
            np.save(output_file, audio_codes)

            # Update progress
            progress["completed"].append(ytid)
            session_encoded += 1

            # Save progress every 10 files
            if session_encoded % 10 == 0:
                progress["total_processed"] = len(progress["completed"])
                progress["total_time_seconds"] += (time.time() - start_time)
                save_progress(progress)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            errors.append({"ytid": ytid, "error": str(e)})
            progress["failed"].append(ytid)

    # Final save
    session_time = time.time() - start_time
    progress["total_time_seconds"] += session_time
    progress["total_processed"] = len(progress["completed"])
    save_progress(progress)

    # Summary
    print("\n" + "=" * 60)
    if interrupted:
        print("⏸️  ENCODING PAUSED")
    else:
        print("✅ ENCODING COMPLETE")

    print(f"\n📊 Session Summary:")
    print(f"   Files encoded: {session_encoded}")
    print(f"   Session time: {timedelta(seconds=int(session_time))}")
    print(f"   Total encoded: {len(progress['completed'])}/{total_files}")
    print(f"   Errors: {len(errors)}")

    if len(progress['completed']) < total_files:
        print(f"\n💡 Run this script again to continue encoding remaining files")

    # Save metadata for all encoded files
    if len(progress['completed']) > 0:
        metadata = {}
        for ytid in progress['completed']:
            metadata[ytid] = {
                "encoded_file": str(output_dir / f"{ytid}.npy")
            }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n📝 Metadata saved to {metadata_file}")

if __name__ == "__main__":
    main()