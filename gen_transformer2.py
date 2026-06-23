import os, sys, math, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
from miditok import TokSequence
from src.config import CHECKPOINTS_DIR, GENERATED_MIDIS_DIR, PROCESSED_DIR
from src.preprocessing.tokenizer import get_remi_tokenizer
from src.models.transformer import MusicTransformer

device = torch.device("cpu")
tokenizer = get_remi_tokenizer()
vocab_size = len(tokenizer)

model = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "task3_transformer_fixed.pth", map_location=device))
model.eval()

SEQ_LEN = 256
all_tokens = np.load(PROCESSED_DIR / "transformer_tokens.npy")

def forward_causal(model, x):
    seq_len = x.size(1)
    cmask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
    e = model.embedding(x) * math.sqrt(model.d_model)
    e = model.positional_encoding(e)
    e = model.transformer_encoder(e, mask=cmask)
    return model.output_projection(e)

def generate(seed, max_len=350, temperature=0.9, top_k=16):
    tokens = list(seed)
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([tokens[-SEQ_LEN:]], dtype=torch.long, device=device)
            logits = forward_causal(model, x)[0, -1, :] / temperature
            tk_logits, tk_idx = torch.topk(logits, top_k)
            probs = F.softmax(tk_logits, dim=-1)
            nxt = tk_idx[torch.multinomial(probs, 1)].item()
            tokens.append(nxt)
    return tokens

def render_wav(midi_path, wav_path, fs=22050):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=fs)
    if np.abs(audio).max() < 1e-9:
        print("   (silent)"); return False
    audio = audio / (np.abs(audio).max() + 1e-9)
    a16 = (audio * 32767 * 0.9).astype(np.int16)
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(a16.tobytes())
    return True

rng = np.random.default_rng(7)
for i in range(3):
    start = int(rng.integers(0, len(all_tokens) - 32))
    seed = all_tokens[start:start + 16].tolist()   # prime with 16 REAL tokens
    toks = generate(seed, max_len=350, temperature=0.9, top_k=16)
    mid = GENERATED_MIDIS_DIR / f"task3_FIXED2_{i+1}.mid"
    try:
        midi = tokenizer([TokSequence(ids=toks)])
        midi.dump_midi(str(mid))
        pm = pretty_midi.PrettyMIDI(str(mid))
        nnotes = sum(len(inst.notes) for inst in pm.instruments)
        dur = pm.get_end_time()
        ok = render_wav(mid, GENERATED_MIDIS_DIR / f"task3_FIXED2_{i+1}.wav")
        rate = nnotes / dur if dur > 0 else 0
        print(f"sample {i+1}: {nnotes} notes, {dur:.1f}s ({rate:.1f} notes/sec){'' if ok else ' [silent]'}")
    except Exception as e:
        print(f"sample {i+1} failed: {e}")
print("Done - listen to task3_FIXED2_1/2/3.wav")