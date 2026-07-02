# 🎹 Unsupervised Neural Networks for Multi-Genre Symbolic Music Generation

Four progressively complex deep-learning models that learn to generate piano music from the **MAESTRO** dataset — an LSTM Autoencoder, a Variational Autoencoder, an autoregressive Transformer, and an RLHF-fine-tuned Transformer.

Beyond building the models, the core of this project was **diagnosing and fixing four distinct failure modes** that were silently degrading the generated output — turning models that produced silence and noise into ones that generate coherent, measurably better music.

---

## 🔧 Engineering Highlights

Each model trained without throwing errors, yet produced degenerate output (silence, noise, or collapse). The real work was diagnosing *why* — and fixing it. Every fix below is backed by a measured before/after.

### 1 · Variational Autoencoder — Posterior Collapse
- **Symptom:** every generated sample was completely silent.
- **Diagnosis:** the KL term overwhelmed the (tiny) reconstruction loss, driving KL → 0. Confirmed by decoding three *different* random latent vectors and getting **identical** output — the decoder had learned to ignore the latent space entirely.
- **Fix:** free-bits KL regularization (a per-dimension KL floor) that keeps the latent informative.
- **Result:** output diversity rose from **~0.001 → ~0.094** (≈90×). The model now generates varied, musical pieces.

### 2 · Transformer — Missing Causal Mask + Insufficient Data
- **Symptom:** generated 12–22 *minute* sequences containing only ~20 notes (effectively silence).
- **Diagnosis:** two issues — (a) the attention layer applied **no causal mask**, so during training each position could attend to *future* tokens (the model "cheated"), creating a train/inference mismatch that broke autoregressive generation; (b) the model had only been trained on **50 files**.
- **Fix:** added a causal (look-ahead) attention mask and retrained on substantially more data.
- **Result:** validation perplexity dropped **51 → 17**; note density went from ~0.1 notes/sec (silence) to **8–15 notes/sec** (musical), with distinct output per sample.

### 3 · RLHF — REINFORCE Instability (Reward Collapse)
- **Symptom:** reward climbed for ~40 episodes, then collapsed; training loss swung between ±300.
- **Diagnosis:** the policy-gradient loss used *raw reward × summed log-probabilities* with no baseline — extremely high variance, leading to destabilizing updates.
- **Fix:** subtract a running-average reward baseline and length-normalize the loss.
- **Result:** loss swings shrank from **±300 → ±3.5**; reward rose and held stable with no collapse.

### 4 · LSTM Autoencoder — Invalid Generation Procedure
- **Symptom:** outputs were ~0.1-second silent bursts.
- **Diagnosis:** the code decoded *random* latent vectors and applied a threshold (0.4) above the model's maximum output probability (~0.35), cutting every note. An autoencoder must reconstruct *encoded real input* — not decode noise.
- **Fix:** reconstruct real piano-roll windows and use adaptive top-k note selection.
- **Result:** now reconstructs full-length musical structure (lossy, as expected of a tight compression bottleneck).

---

## 📊 Results at a Glance

| Task | Model | Metric | Before | After |
|------|-------|--------|--------|-------|
| 1 | LSTM Autoencoder | reconstruction | 0.1s silent bursts | reconstructs musical structure |
| 2 | Variational Autoencoder | latent diversity | ~0.001 (collapsed) | **~0.094** |
| 3 | Transformer | validation perplexity | 51 | **17** |
| 4 | RLHF Transformer | training loss variance | ±300 (collapsed) | **±3.5 (stable)** |

---

## 🧠 Project Overview

The project implements a progression of unsupervised generative models of increasing complexity, all trained on the **MAESTRO v3.0.0** classical-piano dataset. Music is represented two ways:

- **Tasks 1–2:** binarized **piano-roll matrices** — 128 time-steps × 88 keys, at 16 frames/second.
- **Tasks 3–4:** discrete **REMI token sequences** (via MidiTok) — treating music as a language-modeling problem.

| # | Model | What it demonstrates |
|---|-------|----------------------|
| 1 | **LSTM Autoencoder** | Unsupervised compression & reconstruction of musical structure through a latent bottleneck. Uses a custom Binary Focal Loss to handle extreme note sparsity. |
| 2 | **Variational Autoencoder** | Probabilistic latent modeling for *generation* (not just reconstruction), with free-bits regularization to keep the latent space expressive. |
| 3 | **Autoregressive Transformer** | Discrete sequence modeling with a causal decoder over REMI tokens; generates music token-by-token with top-k / temperature sampling. |
| 4 | **RLHF Fine-Tuning** | REINFORCE policy-gradient fine-tuning of the Transformer against a heuristic musical reward — the technique behind modern LLM alignment, applied to symbolic music. |

**Baselines:** Random and first-order Markov-chain generators provide quantitative lower bounds for the learned models.

---

## 🎧 Generated Samples

Audio (`.wav`) and MIDI (`.mid`) samples are in [`outputs/generated_midis/`](outputs/generated_midis/):

- `task1_BETTER_RECON_*` — autoencoder reconstructions vs. originals
- `task2_vae_MUSICAL_*` — VAE generations
- `task3_FIXED2_*` — Transformer generations
- `markov_baseline.mid`, `random_baseline.mid` — baseline comparisons

Training curves are in [`outputs/plots/`](outputs/plots/) (perplexity, reward, loss).

---

## 🗂️ Data Representation

| Parameter | Value |
|-----------|-------|
| Dataset | MAESTRO v3.0.0 (MIDI-only) |
| Pitch range | 88 keys (A0–C8) |
| Frame rate | 16 frames / second |
| Window length | 128 time-steps (~8 seconds) |
| Sparsity filter | windows with <2% active cells discarded |
| Tokenization (Tasks 3–4) | REMI (MidiTok) |

The preprocessing pipeline rebuilds the entire training set directly from the public MAESTRO MIDI files, so the project is **fully reproducible** from raw data.

---

## 🚀 Setup & Usage

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download MAESTRO (MIDI-only)**
Download `maestro-v3.0.0-midi.zip` from the [MAESTRO dataset page](https://magenta.tensorflow.org/datasets/maestro), and place the `maestro-v3.0.0.csv` and year folders into `data/raw_maestro/`.

**3. Build the dataset**
```bash
python build_data.py
```

**4. Train / generate** (per task)
```bash
python train_vae_fixed.py          # Task 2: train VAE
python gen_musical.py              # Task 2: generate
python train_transformer_fast.py   # Task 3: train Transformer
python gen_transformer2.py         # Task 3: generate
python train_rlhf_fixed.py         # Task 4: RLHF fine-tuning
```

---

## 🛠️ Tech Stack

**PyTorch** · **pretty_midi** · **MidiTok (REMI)** · **NumPy** · **Matplotlib** · **MAESTRO v3.0.0**

---

## 📁 Repository Structure

```
├── src/
│   ├── models/          # Autoencoder, VAE, Transformer, baselines
│   ├── preprocessing/   # MIDI parsing, piano-roll, REMI tokenizer
│   ├── training/        # training routines
│   ├── generation/      # sampling & MIDI export
│   └── evaluation/      # reward & metrics
├── notebooks/           # EDA + per-task development notebooks
├── outputs/
│   ├── checkpoints/     # trained model weights
│   ├── generated_midis/ # generated audio & MIDI
│   └── plots/           # training curves
└── build_data.py        # rebuild dataset from raw MAESTRO
```

---

## ⚠️ Notes & Limitations

- Models were trained on **CPU** with data subsets for accessibility; scaling data and compute would further improve fidelity. The goal here is correct, working implementations with measurable results — not state-of-the-art audio quality.
- The autoencoder reconstruction is **lossy by design** (information is compressed through a tight latent bottleneck).
- The RLHF reward is a **programmatic heuristic** (a proxy for human feedback) that rewards melodic variety and valid structure while penalizing repetition — not actual human ratings.
- MAESTRO is a widely-studied dataset; this project's contribution is **engineering rigor and breadth** across four architectures, not dataset novelty.