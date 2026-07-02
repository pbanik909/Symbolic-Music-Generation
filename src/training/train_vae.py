"""Training routine for the LSTM-VAE with free-bits KL (prevents posterior collapse)."""
import torch


def train_vae(model, loader, epochs=30, lr=1e-3, kl_warmup=8, free_bits=0.5, device="cpu"):
    """
    Train the VAE with teacher forcing + free-bits KL regularization.

    A per-dimension KL floor (`free_bits`) keeps each latent dimension carrying at
    least that many nats of information. This stops the KL term from collapsing the
    latent space to the prior -- the failure mode that produced silent output.
    """
    from src.utils.losses import BinaryFocalLoss
    recon_criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def tf_forward(batch):
        B, T, P = batch.shape
        mu, log_var = model.encode(batch)
        z = model.reparameterize(mu, log_var)
        h0 = model.decoder_linear(z).unsqueeze(0).repeat(model.num_layers, 1, 1).contiguous()
        c0 = torch.zeros_like(h0)
        start = torch.zeros(B, 1, P, device=batch.device)
        dec_in = torch.cat([start, batch[:, :-1, :]], dim=1)
        dec_out, _ = model.decoder_lstm(dec_in, (h0, c0))
        return model.output_layer(dec_out), mu, log_var

    for epoch in range(epochs):
        model.train()
        beta = min(1.0, epoch / kl_warmup)
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, log_var = tf_forward(batch)
            recon = recon_criterion(logits, batch)
            kl_per_dim = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean(dim=0)
            kl = torch.clamp(kl_per_dim, min=free_bits).sum()
            (recon + beta * kl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
