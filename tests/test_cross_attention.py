# tests/test_cross_attention.py

import torch
from models.components.cross_attention_gpt import (
    CrossAttentionGPT,
    CrossAttentionGPTConfig,
)


def test_cross_attention_gpt():
    # Create a small model for testing
    config = CrossAttentionGPTConfig(
        block_size=128, vocab_size=100, n_layer=2, n_head=4, n_embd=64, embedding_dim=32
    )

    model = CrossAttentionGPT(config)

    # Test forward pass with embeddings
    batch_size = 2
    seq_len = 10

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    embeddings = torch.randn(batch_size, config.embedding_dim)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(idx, targets=targets, embeddings=embeddings)

    # Check shapes
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss.item() > 0  # Just make sure loss is computed

    # Test generation
    output = model.generate(idx[:, :1], max_new_tokens=5, embeddings=embeddings)
    assert output.shape == (batch_size, 6)  # 1 + 5 new tokens

    print("Cross-attention GPT model test passed")


if __name__ == "__main__":
    test_cross_attention_gpt()
    print("All tests passed!")
