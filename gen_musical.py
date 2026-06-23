import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pretty_midi
from src.config import CHECKPOINTS_DIR, GENERATED_MIDIS_DIR, START_PITCH, FS
from src.models.vae import LSTM_VAE

device = torch.device("cpu")
model = LSTM_VAE().to(device)
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "task2_lstm_vae_fixed.pth", map_location=device))
model.eval()

def topk_to_midi(probs, path, k=4, min_prob=0.5):
    T, P = probs.shape
    binary = np.zeros((T, P), dtype=int)
    for t in range(T):
        idx = np.argsort(probs[t])[-k:]
        for pi in idx:
            if probs[t, pi] >= min_prob:
                binary[t, pi] = 1
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    fd = 1.0 / FS
    for pi in range(P):
        pitch = pi + START_PITCH
        active = np.where(binary[:, pi] == 1)[0]
        if len(active) == 0: continue
        groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
        for g in groups:
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=int(pitch), start=float(g[0]*fd), end=float((g[-1]+1)*fd)))
    midi.instruments.append(piano)
    midi.write(str(path))
    return int(binary.sum())

torch.manual_seed(1)
with torch.no_grad():
    for i in range(3):
        z = torch.randn(1, model.latent_dim)
        probs = torch.sigmoid(model.decode(z, seq_len=128)).numpy()[0]
        out = GENERATED_MIDIS_DIR / f"task2_vae_MUSICAL_{i+1}.mid"
        n = topk_to_midi(probs, out, k=4, min_prob=0.5)
        print(f"sample {i+1}: {n} notes ({100*n/(128*88):.1f}% density) -> {out.name}")
print("Done - listen to the MUSICAL midis.")
