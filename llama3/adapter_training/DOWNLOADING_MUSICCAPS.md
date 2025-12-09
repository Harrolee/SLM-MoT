# Downloading MusicCaps Audio Files

## Quick Start

### Step 1: Download a Subset

Start with a small subset to test:

```bash
cd /Users/lee/fun/behavior_cloning/llama3/adapter_training
source ../venv/bin/activate
python download_musiccaps_subset.py --limit 500 --num-proc 4
```

**What this does:**
- Downloads 500 audio clips from YouTube
- Saves them as WAV files in `./musiccaps_audio/`
- Uses 4 parallel processes (adjust based on your system)
- Takes ~30-60 minutes for 500 samples

### Step 2: Train with Downloaded Audio

Once downloaded, restart training:

```bash
python train_adapter.py
```

The training script will automatically detect downloaded audio files and use them!

## Options

### Download More Samples

```bash
# Download 1000 samples
python download_musiccaps_subset.py --limit 1000

# Download ALL samples (~5,521)
python download_musiccaps_subset.py --limit None
```

### Adjust Parallelism

```bash
# More parallel downloads (faster but more CPU/network)
python download_musiccaps_subset.py --limit 500 --num-proc 8

# Fewer parallel downloads (slower but more stable)
python download_musiccaps_subset.py --limit 500 --num-proc 2
```

### Custom Directory

```bash
python download_musiccaps_subset.py --data-dir ./my_audio_files --limit 500
```

## What Gets Downloaded

- **Format**: WAV files
- **Sampling Rate**: 32kHz (matches MusicGen)
- **Location**: `./musiccaps_audio/` (or custom `--data-dir`)
- **Naming**: `{ytid}.wav` (e.g., `-0Gj8-vB1q4.wav`)

## Expected Behavior

### During Download

You'll see progress as files download:
- Some videos may fail (deleted, region-locked, etc.)
- Success rate is typically 80-90%
- Failed downloads are skipped

### During Training

The training script will:
1. Check for `./musiccaps_audio/` directory
2. If found, use downloaded audio files
3. If not found, fall back to streaming/dummy data

## Troubleshooting

### "yt-dlp not found"
```bash
pip install yt-dlp
```

### "ffmpeg not found"
```bash
brew install ffmpeg
```

### Downloads Failing

- Some videos are deleted/region-locked (normal)
- Try reducing `--num-proc` if getting rate-limited
- Check internet connection

### Out of Disk Space

- Each sample is ~640KB
- 500 samples ≈ 320MB
- 5,521 samples ≈ 3.5GB
- Monitor disk space!

## Recommendation

**Start Small:**
1. Download 500 samples first (`--limit 500`)
2. Test training with real data
3. If it works well, download more
4. Scale up to full dataset if needed

**Time Estimates:**
- 500 samples: ~30-60 minutes
- 1000 samples: ~1-2 hours  
- Full dataset: ~5-10 hours

## Next Steps

1. **Download subset**: `python download_musiccaps_subset.py --limit 500`
2. **Wait for download** (grab coffee ☕)
3. **Restart training**: `python train_adapter.py`
4. **Monitor loss**: Should improve with real data!

