#!/bin/bash
# Lambda Labs Setup Script for MoT Training
# Run this after SSH'ing into your Lambda Labs instance

set -e  # Exit on error

echo "🚀 Setting up Lambda Labs environment for MoT training..."

# Check GPU availability
echo "🔍 Checking GPU..."
nvidia-smi

# Clone repository (adjust URL as needed)
echo "📦 Cloning repository..."
git clone https://github.com/YOUR_USERNAME/behavior_cloning.git
cd behavior_cloning/llama3/adapter_training

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements_lambda.txt

# Login to HuggingFace (you'll need to provide your token)
echo "🤗 Login to HuggingFace..."
huggingface-cli login

# Login to Weights & Biases (optional)
echo "📊 Login to W&B (optional, press Ctrl+C to skip)..."
wandb login || true

# Test dataset access
echo "✅ Testing dataset access..."
python -c "
from datasets import load_dataset
ds = load_dataset('LeeHarrold/musiccaps-mot-tokens', split='train')
print(f'✅ Successfully loaded {len(ds)} training samples!')
"

# Create directories
mkdir -p checkpoints
mkdir -p logs

echo "✅ Setup complete! Ready to train."
echo ""
echo "To start training, run:"
echo "  source venv/bin/activate"
echo "  python train_lambda.py"
echo ""
echo "For background training with logging:"
echo "  nohup python train_lambda.py > logs/training.log 2>&1 &"
echo "  tail -f logs/training.log  # To monitor progress"