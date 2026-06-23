import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pretty_midi
from src.config import CHECKPOINTS_DIR, GENERATED_MIDIS_DIR, START_PITCH, FS
from src.models.vae import LSTM_VAE

device = torch.device("cpu")
model = LSTM_VAE().to(device)
ckpt = CHECKPOINTS_DIR / "task2_lstm_vae_fixed.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print("Loaded:", ckpt)

def matrix_to_midi(matrix, path, threshold):
    binary = (matrix > threshold).astype(int)
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    fd = 1.0 / FS
    for pi in range(binary.shape[1]):
        pitch = pi + START_PITCH
        active = np.where(binary[:, pi] == 1)[0]
        if len(active) == 0: continue
        groups = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
        for g in groups:
            piano.notes.append(pretty_midi.Note(velocity=80, pitch=int(pitch), start=float(g[0]*fd), end=float((g[-1]+1)*fd)))
    midi.instruments.append(piano)
    midi.write(str(path))

torch.manual_seed(0)
outs = []
with torch.no_grad():
    for i in range(3):
        z = torch.randn(1, model.latent_dim)
        probs = torch.sigmoid(model.decode(z, seq_len=128)).numpy()[0]
        outs.append(probs)
        print(f"z{i+1}: max={probs.max():.3f} mean={probs.mean():.4f} | notes>0.2={(probs>0.2).sum()} >0.3={(probs>0.3).sum()} >0.4={(probs>0.4).sum()}")

d = (np.abs(outs[0]-outs[1]).mean() + np.abs(outs[0]-outs[2]).mean())/2
print(f"DIVERSITY across z: {d:.5f}  (original was ~0.001; >0.003 = working)")

th = 0.3
for i, probs in enumerate(outs):
    out = GENERATED_MIDIS_DIR / f"task2_vae_FIXED_{i+1}.mid"
    matrix_to_midi(probs, out, th)
    print(f"saved {out.name}  ({(probs>th).sum()} active cells at thresh {th})")
print("Done - open the FIXED midis to listen.")
