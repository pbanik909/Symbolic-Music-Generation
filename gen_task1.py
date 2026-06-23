import os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pretty_midi
from src.config import CHECKPOINTS_DIR, GENERATED_MIDIS_DIR, PROCESSED_DIR, START_PITCH, FS
from src.models.autoencoder import LSTMAutoencoder

device = torch.device("cpu")
model = LSTMAutoencoder().to(device)
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "task1_lstm_ae.pth", map_location=device))
model.eval()

data = np.load(PROCESSED_DIR / "maestro_validation_windows.npy")
densities = data.reshape(len(data), -1).sum(axis=1)
idx = np.argsort(densities)[-200:]
rng = np.random.default_rng(0)
chosen = rng.choice(idx, 3, replace=False)

def binary_to_midi(binary, path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    fd = 1.0 / FS
    for pi in range(binary.shape[1]):
        active = np.where(binary[:, pi] == 1)[0]
        if len(active) == 0:
            continue
        for g in np.split(active, np.where(np.diff(active) != 1)[0] + 1):
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=int(pi + START_PITCH),
                                                start=float(g[0] * fd), end=float((g[-1] + 1) * fd)))
    midi.instruments.append(piano)
    midi.write(str(path))

def topk(probs, k=5, min_prob=0.2):
    out = np.zeros_like(probs, dtype=int)
    for t in range(probs.shape[0]):
        for pi in np.argsort(probs[t])[-k:]:
            if probs[t, pi] >= min_prob:
                out[t, pi] = 1
    return out

def render_wav(midi_path, wav_path, fs=22050):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=fs)
    audio = audio / (np.abs(audio).max() + 1e-9)
    a16 = (audio * 32767 * 0.9).astype(np.int16)
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(a16.tobytes())

for n, ci in enumerate(chosen):
    original = data[ci]
    with torch.no_grad():
        z = model.encode(torch.tensor(original[None], dtype=torch.float32))
        recon = torch.sigmoid(model.decode(z, 128)).numpy()[0]
    recon_bin = topk(recon, k=5, min_prob=0.2)
    tp = int((recon_bin * (original > 0.5)).sum())
    total = int((original > 0.5).sum())
    om = GENERATED_MIDIS_DIR / f"task1_ORIGINAL_{n+1}.mid"
    rm = GENERATED_MIDIS_DIR / f"task1_RECON_{n+1}.mid"
    binary_to_midi((original > 0.5).astype(int), om)
    binary_to_midi(recon_bin, rm)
    render_wav(om, GENERATED_MIDIS_DIR / f"task1_ORIGINAL_{n+1}.wav")
    render_wav(rm, GENERATED_MIDIS_DIR / f"task1_RECON_{n+1}.wav")
    print(f"pair {n+1}: original {total} notes -> recon recovered {tp} ({100*tp/total:.0f}%)")
print("Done - compare task1_ORIGINAL_N.wav vs task1_RECON_N.wav")