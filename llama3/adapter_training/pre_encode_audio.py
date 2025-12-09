"""
Pre-encode all MusicCaps audio files to tokens.
This eliminates the computation graph connection issue during training.
"""
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def pre_encode_audio_files(audio_dir="./musiccaps_audio", output_dir="./pre_encoded_audio", max_files=None):
    """
    Pre-encode all audio files in audio_dir and save tokens to output_dir.
    
    Args:
        audio_dir: Directory containing .wav files
        output_dir: Directory to save encoded tokens
        max_files: Maximum number of files to encode (None = all)
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all audio files
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"📦 Found {len(audio_files)} audio files to encode")
    
    # Load MusicGen model and processor
    print("🔧 Loading MusicGen model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    musicgen.eval()
    musicgen.to(device)
    
    # Freeze all parameters
    for param in musicgen.parameters():
        param.requires_grad = False
    
    # Metadata file to store file mappings
    metadata = {}
    
    print(f"🎵 Encoding audio files (device: {device})...")
    successful = 0
    failed = 0
    
    for idx, audio_file in enumerate(tqdm(audio_files, desc="Encoding")):
        try:
            # Load audio
            import soundfile as sf
            audio_data, sampling_rate = sf.read(str(audio_file))
            
            # Ensure mono (take first channel if stereo)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # Ensure float32
            audio_data = audio_data.astype(np.float32)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample if needed (MusicGen expects 32kHz)
            if sampling_rate != 32000:
                from scipy.signal import resample_poly
                audio_data = resample_poly(audio_data, 32000, sampling_rate)
                sampling_rate = 32000
            
            # Process audio - ensure it's a list of arrays
            inputs = processor(
                audio=[audio_data],
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # Encode to tokens
            with torch.no_grad():
                encoder_outputs = musicgen.audio_encoder(inputs.input_values)
                audio_codes = encoder_outputs.audio_codes
                
                # Handle list/tensor formats
                if isinstance(audio_codes, list):
                    audio_codes = torch.stack(audio_codes)
                elif isinstance(audio_codes, torch.Tensor):
                    pass
                else:
                    raise ValueError(f"Unexpected audio_codes type: {type(audio_codes)}")
                
                # Ensure correct shape: [batch, n_codebooks, seq_len]
                while audio_codes.dim() > 3:
                    audio_codes = audio_codes.squeeze()
                
                if audio_codes.dim() == 2:
                    # [n_codebooks, seq_len] -> [1, n_codebooks, seq_len]
                    audio_codes = audio_codes.unsqueeze(0)
                
                # Move to CPU and convert to numpy for storage
                audio_codes = audio_codes.cpu().detach().numpy()
            
            # Save as numpy file
            output_file = output_dir / f"{audio_file.stem}.npy"
            np.save(output_file, audio_codes)
            
            # Store metadata
            metadata[audio_file.stem] = {
                "original_file": str(audio_file),
                "encoded_file": str(output_file),
                "shape": list(audio_codes.shape),
                "n_codebooks": audio_codes.shape[1],
                "seq_len": audio_codes.shape[2]
            }
            
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Failed to encode {audio_file.name}: {e}")
            failed += 1
            continue
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Encoding complete!")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")
    print(f"   - Metadata saved to: {metadata_file}")
    print(f"   - Encoded tokens saved to: {output_dir}")
    
    return metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-encode MusicCaps audio files")
    parser.add_argument("--audio_dir", type=str, default="./musiccaps_audio", 
                       help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./pre_encoded_audio",
                       help="Directory to save encoded tokens")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to encode (for testing)")
    
    args = parser.parse_args()
    
    pre_encode_audio_files(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        max_files=args.max_files
    )

