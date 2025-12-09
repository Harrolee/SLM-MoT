# Next Steps: Retrain and Test

## Current Status

✅ **Fixed Issues:**
- Vocabulary expansion (128256 → 130304)
- Output layer now trainable (was frozen before!)
- Architecture documented

⚠️ **Old Checkpoints:**
- `adapter_step_100.pt`, `adapter_step_200.pt`, `adapter_step_300.pt`, `adapter_final.pt`
- These were trained with **output layer frozen** - won't work well for audio generation
- Need to retrain with the fix!

## Step 1: Retrain Adapters (With Output Layer Trainable)

The training script is now fixed. Run:

```bash
cd /Users/lee/fun/behavior_cloning/llama3/adapter_training
source ../venv/bin/activate
python train_adapter.py 2>&1 | tee training_new.log
```

**What's different now:**
- ✅ Output layer is trainable (~260M params)
- ✅ New audio token positions (128256-130303) will be trained
- ✅ Model can actually learn to predict audio tokens!

**Expected:**
- Training will take longer (more parameters)
- Checkpoints will be larger (includes output layer weights)
- Loss should decrease as model learns audio tokens

## Step 2: Run Inference with New Checkpoints

After training completes, test inference:

```bash
cd /Users/lee/fun/behavior_cloning/llama3/adapter_training
source ../venv/bin/activate
python inference_with_adapters.py
```

**What should happen:**
- Loads trained adapters + output layer weights
- Generates audio tokens autoregressively
- Decodes to `.wav` file
- You can listen to the generated audio! 🎵

## Step 3: Verify It Works

1. **Check checkpoint size**: Should be larger (~1GB+ instead of 768MB)
2. **Check training logs**: Loss should decrease
3. **Run inference**: Should generate audio tokens successfully
4. **Listen to audio**: `mot_generated_audio.wav` should have actual audio (not silence/random noise)

## Optional: Clean Up Old Checkpoints

If you want to save space, you can remove old checkpoints:

```bash
# Backup old checkpoints first!
mkdir -p checkpoints_old
mv checkpoints/adapter_step_*.pt checkpoints_old/
mv checkpoints/adapter_final.pt checkpoints_old/
```

## Troubleshooting

### If training is too slow:
- Reduce `max_steps` in `train_adapter.py`
- Use smaller batch size
- Train on GPU if available

### If inference fails:
- Check that checkpoint includes `output.weight` in `adapter_state_dict`
- Verify vocabulary expansion happened correctly
- Check that audio token range is correct (128256-130303)

## Summary

1. 🔄 **Retrain** adapters with output layer trainable
2. 🎵 **Run inference** with new checkpoints  
3. 🎧 **Listen** to generated audio!

Ready to retrain? Let's go! 🚀

