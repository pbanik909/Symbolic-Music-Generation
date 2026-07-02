"""Training routine for the autoregressive Transformer (with causal masking)."""
import math
import torch
import torch.nn as nn


def causal_forward(model, x):
    """Forward pass with a causal mask so each position attends only to previous tokens."""
    seq_len = x.size(1)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
    e = model.embedding(x) * math.sqrt(model.d_model)
    e = model.positional_encoding(e)
    e = model.transformer_encoder(e, mask=mask)
    return model.output_projection(e)


def train_transformer(model, loader, vocab_size, epochs=6, lr=5e-4, device="cpu"):
    """
    Train the Transformer for next-token prediction with a causal attention mask.

    The causal mask matches the autoregressive generation regime. Without it, the
    model attends to future tokens during training and then fails at generation time.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = causal_forward(model, x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
