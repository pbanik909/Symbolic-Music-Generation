import os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pretty_midi
from src.config import GENERATED_MIDIS_DIR

fs = 22050
for i in range(1, 4):
    midi_path = GENERATED_MIDIS_DIR / f"task2_vae_MUSICAL_{i}.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=fs)
    audio = audio / (np.abs(audio).max() + 1e-9)
    audio_int16 = (audio * 32767 * 0.9).astype(np.int16)
    wav_path = GENERATED_MIDIS_DIR / f"task2_vae_MUSICAL_{i}.wav"
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(audio_int16.tobytes())
    print(f"rendered {wav_path.name} ({len(audio)/fs:.1f}s)")
print("Done - find the .wav files and double-click to play.")
