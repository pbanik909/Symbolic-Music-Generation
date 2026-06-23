import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import CHECKPOINTS_DIR, PLOTS_DIR
from src.preprocessing.tokenizer import get_remi_tokenizer
from src.models.transformer import MusicTransformer
from src.evaluation.metrics import calculate_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = get_remi_tokenizer()
vocab_size = len(tokenizer)

# Start from the FIXED transformer (not the old broken one)
model = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
model.load_state_dict(torch.load(CHECKPOINTS_DIR / "task3_transformer_fixed.pth", map_location=device))
print("Loaded FIXED transformer as RLHF starting point")

SEQ_LEN = 256
EPISODES = 60
MAX_LEN = 96
LR = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def forward_causal(model, x):
    L = x.size(1)
    m = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
    e = model.embedding(x) * math.sqrt(model.d_model)
    e = model.positional_encoding(e)
    e = model.transformer_encoder(e, mask=m)
    return model.output_projection(e)

def episode(seed, max_len):
    model.train()
    toks = seed.tolist()
    lps = []
    for _ in range(max_len):
        x = torch.tensor([toks[-SEQ_LEN:]], dtype=torch.long, device=device)
        logits = forward_causal(model, x)[0, -1, :]
        d = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        a = d.sample()
        lps.append(d.log_prob(a))
        toks.append(a.item())
    return toks, torch.stack(lps)

out_path = CHECKPOINTS_DIR / "task4_transformer_rlhf_fixed.pth"
rewards = []
baseline = 0.0
print("Starting STABLE RLHF (baseline + normalized)... reward should rise and NOT collapse.")
t0 = time.time()
for ep in range(EPISODES):
    optimizer.zero_grad()
    seed = torch.tensor([np.random.randint(5, vocab_size)])
    toks, lps = episode(seed, MAX_LEN)
    r = calculate_reward(toks, tokenizer)
    rewards.append(r)
    baseline = 0.9 * baseline + 0.1 * r          # running baseline
    loss = -(r - baseline) * lps.mean()          # length-normalized advantage
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if (ep + 1) % 10 == 0:
        avg = np.mean(rewards[-10:])
        print(f"Episode {ep+1}/{EPISODES} | avg reward (last 10): {avg:.2f} | loss: {loss.item():.3f} | baseline: {baseline:.2f} ({time.time()-t0:.0f}s)")
    if (ep + 1) % 20 == 0:
        torch.save(model.state_dict(), out_path)

torch.save(model.state_dict(), out_path)
# reward plot
plt.figure(figsize=(10, 5))
sm = [np.mean(rewards[max(0, i-4):i+1]) for i in range(len(rewards))]
plt.plot(rewards, alpha=0.3, label="Raw reward")
plt.plot(sm, linewidth=2, label="Smoothed")
plt.title("Task 4: RLHF Rewards (Fixed - Stable)")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend(); plt.grid(True)
plt.savefig(PLOTS_DIR / "task4_rlhf_rewards_fixed.png", dpi=120, bbox_inches="tight")
print("Saved model ->", out_path)
print("Saved reward plot -> task4_rlhf_rewards_fixed.png")
print(f"Total time: {time.time()-t0:.0f}s")