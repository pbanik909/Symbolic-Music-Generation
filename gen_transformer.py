import os, sys, math, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
from miditok import TokSequence
from src.config import CHECKPOINTS_DIR, GENERATED_MIDIS_DIR
from src.preprocessing.tokenizer import get_remi_tokenizer
from src.models.transformer import MusicTransformer

device = torch.device("cpu")
tokenizer = get_remi_tokenizer()
vocab_size = len(tokenizer)

model = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "task3_transformer_fixed.pth", map_location=device))
model.eval()

SEQ_LEN = 256

def forward_causal(model, x):
    seq_len = x.size(1)
    cmask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
    e = model.embedding(x) * math.sqrt(model.d_model)
    e = model.positional_encoding(e)
    e = model.transformer_encoder(e, mask=cmask)
    return model.output_projection(e)

def generate(seed, max_len=400, temperature=0.95, top_k=20):
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
        print("   (silent)"); return
    audio = audio / (np.abs(audio).max() + 1e-9)
    a16 = (audio * 32767 * 0.9).astype(np.int16)
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(a16.tobytes())

for i in range(3):
    seed = [np.random.randint(5, vocab_size)]
    toks = generate(seed, max_len=400, temperature=0.95, top_k=20)
    mid = GENERATED_MIDIS_DIR / f"task3_FIXED_{i+1}.mid"
    try:
        midi = tokenizer([TokSequence(ids=toks)])
        midi.dump_midi(str(mid))
        pm = pretty_midi.PrettyMIDI(str(mid))
        nnotes = sum(len(inst.notes) for inst in pm.instruments)
        dur = pm.get_end_time()
        render_wav(mid, GENERATED_MIDIS_DIR / f"task3_FIXED_{i+1}.wav")
        print(f"sample {i+1}: {nnotes} notes, {dur:.1f}s -> task3_FIXED_{i+1}.wav")
    except Exception as e:
        print(f"sample {i+1} failed: {e}")
print("Done - listen to task3_FIXED_1/2/3.wav")