"""Latent-space sampling and interpolation for the VAE."""
import numpy as np
import torch


def sample_prior(model, num_samples=1, seq_len=128, device="cpu"):
    """Sample latent vectors from the N(0, 1) prior and decode them into piano-rolls."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        logits = model.decode(z, seq_len=seq_len)
        return torch.sigmoid(logits).cpu().numpy()


def interpolate(model, x1, x2, steps=5, seq_len=128, device="cpu"):
    """Encode two real windows and decode a linear path between their latent means."""
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encode(x1.to(device))
        mu2, _ = model.encode(x2.to(device))
        outputs = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            logits = model.decode(z, seq_len=seq_len)
            outputs.append(torch.sigmoid(logits).cpu().numpy()[0])
        return outputs
