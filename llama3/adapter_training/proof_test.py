#!/usr/bin/env python
"""Definitive test to prove backward graph error is fixed."""

import sys
print("=" * 60)
print("BACKWARD GRAPH ERROR PROOF TEST")
print("=" * 60)
sys.stdout.flush()

# Only test the critical parts
print("\n1. Setting max_steps to 3 for quick test...")
import train_adapter
train_adapter.train.__code__ = train_adapter.train.__code__.replace(
    co_consts=tuple(3 if c == 5 else c for c in train_adapter.train.__code__.co_consts)
)

# Monkey patch to get clearer output
original_train = train_adapter.train

def patched_train():
    """Wrapper to catch and report errors clearly."""
    import torch

    # Patch the training loop to be more verbose
    print("\n2. Starting training with pre-encoded tokens...")
    sys.stdout.flush()

    try:
        # Call original training
        original_train()
        print("\n✅ TRAINING COMPLETED WITHOUT BACKWARD GRAPH ERROR!")
        return
    except RuntimeError as e:
        if "backward through the graph a second time" in str(e):
            print("\n❌ BACKWARD GRAPH ERROR STILL EXISTS!")
            print(f"Error: {e}")
            sys.exit(1)
        else:
            print(f"\n❌ Different error: {e}")
            sys.exit(2)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

# Replace the train function
train_adapter.train = patched_train

# Run the test
print("\n3. Loading model and starting training...")
sys.stdout.flush()
train_adapter.train()