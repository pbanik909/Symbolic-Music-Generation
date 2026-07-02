"""Convert model output (piano-roll probabilities) into MIDI files and audio."""
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
