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
dens = data.reshape(len(data), -1).mean(axis=1)
# pick MEDIUM-density windows (clearer/melodic): 40th-60th percentile
lo, hi = np.percentile(dens, 40), np.percentile(dens, 60)
candidates = np.where((dens >= lo) & (dens <= hi))[0]
rng = np.random.default_rng(3)
chosen = rng.choice(candidates, 3, replace=False)

def binary_to_midi(binary, path):
    midi = pretty_midi.PrettyMIDI(); piano = pretty_midi.Instrument(program=0); fd = 1.0/FS
    for pi in range(binary.shape[1]):
        active = np.where(binary[:, pi] == 1)[0]
        if len(active) == 0: continue
        for g in np.split(active, np.where(np.diff(active) != 1)[0] + 1):
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=int(pi+START_PITCH),
                                                start=float(g[0]*fd), end=float((g[-1]+1)*fd)))
    midi.instruments.append(piano); midi.write(str(path))

def topk_match(probs, target_per_frame, min_prob=0.15):
    out = np.zeros_like(probs, int)
    for t in range(probs.shape[0]):
        k = max(1, int(round(target_per_frame[t])))
        for pi in np.argsort(probs[t])[-k:]:
            if probs[t, pi] >= min_prob:
                out[t, pi] = 1
    return out

def render_wav(midi_path, wav_path, fs=22050):
    pm = pretty_midi.PrettyMIDI(str(midi_path)); audio = pm.synthesize(fs=fs)
    if np.abs(audio).max() < 1e-9: return
    audio = audio/(np.abs(audio).max()+1e-9); a16=(audio*32767*0.9).astype(np.int16)
    with wave.open(str(wav_path),"w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(a16.tobytes())

for n, ci in enumerate(chosen):
    original = (data[ci] > 0.5).astype(int)
    per_frame = original.sum(axis=1)  # match reconstruction density to original
    with torch.no_grad():
        z = model.encode(torch.tensor(data[ci][None], dtype=torch.float32))
        recon = torch.sigmoid(model.decode(z, 128)).numpy()[0]
    recon_bin = topk_match(recon, per_frame, min_prob=0.15)
    tp = int((recon_bin * original).sum()); total = int(original.sum())
    om = GENERATED_MIDIS_DIR / f"task1_BETTER_ORIG_{n+1}.mid"
    rm = GENERATED_MIDIS_DIR / f"task1_BETTER_RECON_{n+1}.mid"
    binary_to_midi(original, om); binary_to_midi(recon_bin, rm)
    render_wav(om, GENERATED_MIDIS_DIR / f"task1_BETTER_ORIG_{n+1}.wav")
    render_wav(rm, GENERATED_MIDIS_DIR / f"task1_BETTER_RECON_{n+1}.wav")
    print(f"pair {n+1}: {total} notes -> recovered {tp} ({100*tp/total:.0f}%)")
print("Done - compare task1_BETTER_ORIG_N.wav vs task1_BETTER_RECON_N.wav")