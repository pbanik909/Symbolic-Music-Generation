"""
fill_stubs.py — populates the empty src/ module files with working code
extracted from the project's fixes. Run once from the project root, then
commit and push.
"""
import os

FILES = {}

# ---------------------------------------------------------------------------
FILES["src/generation/midi_export.py"] = r'''"""Convert model output (piano-roll probabilities) into MIDI files and audio."""
import wave
import numpy as np
import pretty_midi
from src.config import START_PITCH, FS


def matrix_to_midi(binary, path):
    """Write a binary piano-roll (T x P) to a MIDI file, spacing notes by 1/FS."""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    frame_duration = 1.0 / FS
    for pitch_idx in range(binary.shape[1]):
        pitch = pitch_idx + START_PITCH
        active = np.where(binary[:, pitch_idx] == 1)[0]
        if len(active) == 0:
            continue
        for group in np.split(active, np.where(np.diff(active) != 1)[0] + 1):
            start = float(group[0] * frame_duration)
            end = float((group[-1] + 1) * frame_duration)
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=end))
    midi.instruments.append(piano)
    midi.write(str(path))


def topk_to_midi(probs, path, k=4, min_prob=0.2):
    """Keep the top-k most likely notes per timestep (like real polyphony), then export."""
    binary = np.zeros_like(probs, dtype=int)
    for t in range(probs.shape[0]):
        for pi in np.argsort(probs[t])[-k:]:
            if probs[t, pi] >= min_prob:
                binary[t, pi] = 1
    matrix_to_midi(binary, path)
    return int(binary.sum())


def render_wav(midi_path, wav_path, fs=22050):
    """Synthesize a MIDI file to a mono WAV for easy listening. Returns False if silent."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=fs)
    if np.abs(audio).max() < 1e-9:
        return False
    audio = audio / (np.abs(audio).max() + 1e-9)
    a16 = (audio * 32767 * 0.9).astype(np.int16)
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(a16.tobytes())
    return True
'''

# ---------------------------------------------------------------------------
FILES["src/generation/sample_latent.py"] = r'''"""Latent-space sampling and interpolation for the VAE."""
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
'''

# ---------------------------------------------------------------------------
FILES["src/generation/generate_music.py"] = r'''"""Autoregressive generation from the Transformer, using a causal attention mask."""
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
'''

# ---------------------------------------------------------------------------
FILES["src/evaluation/pitch_histogram.py"] = r'''"""Pitch-distribution metrics for evaluating generated music."""
import numpy as np


def pitch_histogram(binary):
    """Return the normalized distribution of note pitches in a piano-roll."""
    counts = binary.sum(axis=0).astype(float)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    return counts / total


def pitch_histogram_distance(a, b):
    """L1 distance between two pitch histograms (lower = more similar)."""
    ha, hb = pitch_histogram(a), pitch_histogram(b)
    return float(np.abs(ha - hb).sum())
'''

# ---------------------------------------------------------------------------
FILES["src/evaluation/rhythm_score.py"] = r'''"""Rhythm and repetition metrics for evaluating generated music."""
import numpy as np


def rhythm_diversity(binary):
    """Fraction of distinct onset patterns across timesteps (higher = more varied)."""
    onsets = [tuple(np.where(binary[t] == 1)[0]) for t in range(binary.shape[0])]
    if not onsets:
        return 0.0
    return len(set(onsets)) / len(onsets)


def repetition_ratio(binary):
    """Fraction of consecutive timesteps that are identical (higher = more repetitive)."""
    if binary.shape[0] < 2:
        return 0.0
    same = sum(np.array_equal(binary[t], binary[t + 1]) for t in range(binary.shape[0] - 1))
    return same / (binary.shape[0] - 1)
'''

# ---------------------------------------------------------------------------
FILES["src/training/train_ae.py"] = r'''"""Training routine for the LSTM Autoencoder."""
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
'''

# ---------------------------------------------------------------------------
FILES["src/training/train_vae.py"] = r'''"""Training routine for the LSTM-VAE with free-bits KL (prevents posterior collapse)."""
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
'''

# ---------------------------------------------------------------------------
FILES["src/training/train_transformer.py"] = r'''"""Training routine for the autoregressive Transformer (with causal masking)."""
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
'''

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for path, content in FILES.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("filled:", path)
    print(f"\nDone - {len(FILES)} files written.")