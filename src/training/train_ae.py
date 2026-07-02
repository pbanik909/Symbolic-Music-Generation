"""Training routine for the LSTM Autoencoder."""
import torch


def train_autoencoder(model, loader, epochs=30, lr=1e-3, device="cpu"):
    """Train the autoencoder to reconstruct piano-roll windows through a latent bottleneck."""
    from src.utils.losses import BinaryFocalLoss
    criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
