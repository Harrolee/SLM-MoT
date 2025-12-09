"""
Download a subset of MusicCaps audio clips from YouTube.
This script downloads audio files locally so we can use them for training.
"""
import sys
import os
from pathlib import Path

# Add download script to path
sys.path.append(os.path.dirname(__file__))
from download_musiccaps import main

def download_subset(
    data_dir: str = "./musiccaps_audio",
    limit: int = 500,  # Start with 500 samples
    sampling_rate: int = 32000,  # Match MusicGen's sampling rate
    num_proc: int = 4,  # Parallel downloads
):
    """
    Download a subset of MusicCaps audio clips.
    
    Args:
        data_dir: Directory to save audio files
        limit: Number of samples to download (None for all)
        sampling_rate: Audio sampling rate (32000 for MusicGen)
        num_proc: Number of parallel download processes
    """
    print(f"🎵 Downloading MusicCaps audio clips...")
    print(f"   - Target directory: {data_dir}")
    print(f"   - Limit: {limit} samples")
    print(f"   - Sampling rate: {sampling_rate} Hz")
    print(f"   - Parallel processes: {num_proc}")
    print()
    
    # Download using the script
    ds = main(
        data_dir=data_dir,
        sampling_rate=sampling_rate,
        limit=limit,
        num_proc=num_proc,
        writer_batch_size=100,
    )
    
    # Count successful downloads by counting files in directory
    # (Can't iterate dataset due to torchcodec requirement for audio decoding)
    data_path = Path(data_dir)
    if data_path.exists():
        successful = len(list(data_path.glob("*.wav")))
    else:
        successful = 0
    
    total = len(ds) if hasattr(ds, '__len__') else "unknown"
    
    print(f"\n✅ Download complete!")
    print(f"   - Total samples processed: {total}")
    print(f"   - Audio files downloaded: {successful}")
    if isinstance(total, int):
        print(f"   - Failed/skipped: {total - successful}")
    print(f"   - Audio files saved to: {data_dir}")
    
    return ds

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download MusicCaps audio subset')
    parser.add_argument('--data-dir', type=str, default='./musiccaps_audio',
                       help='Directory to save audio files')
    parser.add_argument('--limit', type=int, default=500, nargs='?',
                       help='Number of samples to download (omit for all)')
    parser.add_argument('--sampling-rate', type=int, default=32000,
                       help='Audio sampling rate')
    parser.add_argument('--num-proc', type=int, default=4,
                       help='Number of parallel download processes')
    
    args = parser.parse_args()
    
    # Handle limit: if not provided, use None for all samples
    limit = None if args.limit is None else args.limit
    
    download_subset(
        data_dir=args.data_dir,
        limit=limit,
        sampling_rate=args.sampling_rate,
        num_proc=args.num_proc,
    )

