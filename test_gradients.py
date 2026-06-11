#!/usr/bin/env python3

import torch as t
from custom_transformer import DemoTransformer, Config


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("Testing gradient flow...")

    # Create a small model for testing
    cfg = Config(debug=False, d_model=32, n_heads=4, d_head=8, d_mlp=128, n_layers=2, n_ctx=16)
    model = DemoTransformer(cfg)

    # Create dummy input
    tokens = t.randint(
        low=0, high=cfg.d_vocab, size=(2, 8), dtype=t.int64
    )  # batch_size=2, seq_len=8

    # Forward pass
    logits = model(tokens)
    print(f"Logits shape: {logits.shape}")

    # Create dummy target (shifted tokens for language modeling)
    targets = tokens.clone()

    # Compute loss
    loss = t.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1)
    )

    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check if gradients exist
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            print(f"✓ {name} has gradients: grad_norm={param.grad.norm().item():.4f}")
            break

    if has_grads:
        print("✅ Gradient flow is working correctly!")
        return True
    else:
        print("❌ No gradients found - gradient flow is broken!")
        return False


if __name__ == "__main__":
    test_gradient_flow()
