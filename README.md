The **MAESTRO Dataset v3.0.0** was sourced from [Kaggle](https://www.kaggle.com/datasets/alonhaviv/the-maestro-dataset-v3-0-0) and localized to `data\raw_maestro` for processing.

# Unsupervised Neural Network for Multi-Genre Music Generation

This repository contains the implementation for the CSE425/EEE474 course project. The goal of this project is to build unsupervised generative neural networks capable of learning musical representations from the MAESTRO dataset and generating structurally coherent, multi-genre music.

## Project Architecture

This project progressively builds generative sequence models of increasing complexity:

1. **Task 1: LSTM Autoencoder**
   - Learns to compress and reconstruct 128-step binarized piano-roll matrices.
   - Utilizes a custom **Binary Focal Loss** to combat heavy class imbalance (sparsity of musical notes).

2. **Task 2: Variational Autoencoder (VAE)**
   - Replaces the deterministic latent space with a probabilistic one.
   - Utilizes the Reparameterization Trick and **KL-Annealing** to prevent posterior collapse.
   - Generates novel multi-genre proxies and demonstrates smooth latent interpolations between pieces.

3. **Task 3: Autoregressive Transformer**
   - Shifts from continuous matrices to discrete language modeling using **MidiTok (REMI)**.
   - Implements a causal decoder-only Transformer with positional embeddings.
   - Generates multi-minute, structurally coherent compositions using Top-K and Temperature sampling.

4. **Task 4: RLHF (Reinforcement Learning from Human Feedback)**
   - _Bonus Task:_ Fine-tunes the pretrained Transformer using the REINFORCE policy gradient algorithm.
   - Uses a programmatic heuristic reward function to penalize repetitive noise and reward melodic variance and structural grammar.

## Setup and Installation

1. Clone the repository.

2. Install the required dependencies:
   pip install -r requirements.txt

3. Download the MAESTRO Dataset (MIDI-only) & place the maestro-v3.0.0.csv and year folders into data/raw_maestro/

## Execution Pipeline

Execute the Jupyter notebooks in notebooks/ sequentially:

00_eda.ipynb: Analyzes dataset distribution and derives proxy genres.

01_preprocessing.ipynb: Extracts binarized piano-rolls for Tasks 1 & 2.

02_baselines.ipynb: Generates Random and Markov Chain lower bounds.

03_task1_autoencoder.ipynb: Trains LSTM AE.

04_task2_vae.ipynb: Trains VAE and runs latent interpolation.

05_task3_transformer.ipynb: Tokenizes data and trains Transformer.

06_task4_rlhf.ipynb: Fine-tunes the Transformer via RL.

## Generated Samples

All generated .mid files, including baseline comparisons and final compositions, can be found in the "outputs/generated_midis/" directory. Checkpoints are stored in "outputs/checkpoints/" and performance charts in "outputs/plots/"


## Ignored Folder "data" ; size:100GB
![alt text](ignored_folder_structure(data).png)