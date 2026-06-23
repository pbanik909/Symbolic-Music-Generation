import os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pretty_midi
from src.config import CHECKPOINTS_DIR, GENERATED_MIDIS_DIR, PROCESSED_DIR, START_PITCH, FS
from src.models.vae import LSTM_VAE

device = torch.device("cpu")
model = LSTM_VAE().to(device)
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "task2_lstm_vae_fixed.pth", map_location=device))
model.eval()

data = np.load(PROCESSED_DIR / "maestro_validation_windows.npy")
rng = np.random.default_rng(0)
samples = torch.tensor(data[rng.choice(len(data), 40, replace=False)], dtype=torch.float32)
with torch.no_grad():
    mu, _ = model.encode(samples)
mu_np = mu.numpy()
dists = np.linalg.norm(mu_np[:, None, :] - mu_np[None, :, :], axis=2)
i, j = np.unravel_index(np.argmax(dists), dists.shape)
z1, z2 = mu[i:i+1], mu[j:j+1]
print(f"Morphing piece {i} -> piece {j} (latent distance {dists[i,j]:.2f})")

def topk_to_midi(probs, path, k=4, min_prob=0.5):
    T, P = probs.shape
    binary = np.zeros((T, P), dtype=int)
    for t in range(T):
        for pi in np.argsort(probs[t])[-k:]:
            if probs[t, pi] >= min_prob:
                binary[t, pi] = 1
    midi = pretty_midi.PrettyMIDI(); piano = pretty_midi.Instrument(program=0); fd = 1.0/FS
    for pi in range(P):
        active = np.where(binary[:, pi] == 1)[0]
        if len(active) == 0: continue
        for g in np.split(active, np.where(np.diff(active) != 1)[0] + 1):
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=int(pi+START_PITCH), start=float(g[0]*fd), end=float((g[-1]+1)*fd)))
    midi.instruments.append(piano); midi.write(str(path)); return int(binary.sum())

def render_wav(midi_path, wav_path, fs=22050):
    pm = pretty_midi.PrettyMIDI(str(midi_path)); audio = pm.synthesize(fs=fs)
    audio = audio / (np.abs(audio).max() + 1e-9); a16 = (audio*32767*0.9).astype(np.int16)
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(a16.tobytes())

for s, a in enumerate(np.linspace(0, 1, 5)):
    with torch.no_grad():
        z = (1 - a) * z1 + a * z2
        probs = torch.sigmoid(model.decode(z, seq_len=128)).numpy()[0]
    mid = GENERATED_MIDIS_DIR / f"task2_vae_MORPH_{s+1}.mid"
    wav = GENERATED_MIDIS_DIR / f"task2_vae_MORPH_{s+1}.wav"
    n = topk_to_midi(probs, mid); render_wav(mid, wav)
    print(f"step {s+1} (alpha={a:.2f}): {n} notes -> {wav.name}")
print("Done - play MORPH_1 to MORPH_5 in order to hear the style morph.")
