"""Autoregressive generation from the Transformer, using a causal attention mask."""
import math
import torch
import torch.nn.functional as F


def causal_forward(model, x):
    """Forward pass with a causal (look-ahead) mask so each position sees only the past."""
    seq_len = x.size(1)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
    e = model.embedding(x) * math.sqrt(model.d_model)
    e = model.positional_encoding(e)
    e = model.transformer_encoder(e, mask=mask)
    return model.output_projection(e)


def generate(model, seed, max_len=400, temperature=0.9, top_k=16, seq_len=256, device="cpu"):
    """Generate a REMI token sequence with top-k / temperature sampling."""
    model.eval()
    tokens = list(seed)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
            logits = causal_forward(model, x)[0, -1, :] / temperature
            tk_logits, tk_idx = torch.topk(logits, top_k)
            probs = F.softmax(tk_logits, dim=-1)
            nxt = tk_idx[torch.multinomial(probs, 1)].item()
            tokens.append(nxt)
    return tokens
