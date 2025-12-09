# Adapter Training Status

## 🚀 Training Started

Adapter training has been initiated on your MacBook!

### Current Status
- **Process**: Running in background
- **Device**: MPS (Metal Performance Shaders)
- **Checkpoints**: Will be saved to `adapter_training/checkpoints/`
- **Logs**: Check `adapter_training/training.log`

### Training Configuration
- **Epochs**: 3
- **Max Steps**: 1000 (for initial run)
- **Batch Size**: 1 (MacBook-friendly)
- **Save Frequency**: Every 100 steps
- **Log Frequency**: Every 10 steps

### Monitor Training

```bash
# Watch training progress
cd llama3/adapter_training
tail -f training.log

# Check for checkpoints
ls -lh checkpoints/

# Check if process is running
ps aux | grep train_adapter
```

### Expected Output
- Checkpoints saved as: `checkpoints/adapter_step_*.pt`
- Final checkpoint: `checkpoints/adapter_final.pt`
- Each checkpoint contains only adapter weights (67M parameters)

### Next Steps After Training
1. Load trained adapter weights into MoT model
2. Run inference to generate audio tokens
3. Decode tokens → waveform using EnCodec
4. Listen to generated audio!

---

**Note**: Training on MacBook will be slower than GPU, but it's running! 
Check `training.log` periodically to see progress.

