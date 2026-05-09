import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PianoRollDataset(Dataset):
    """
    Dataset for Tasks 1 & 2 (AE and VAE).
    Loads pre-extracted binarized piano roll windows.
    """
    def __init__(self, windows_array):
        # windows_array is expected to be a numpy array of shape (N, 128, 88)
        self.data = torch.tensor(windows_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Autoencoders require the input and target to be the same (X, X)
        x = self.data[idx]
        return x, x

class REMITokenDataset(Dataset):
    """
    Dataset for Task 3 (Transformer).
    Yields token sequences.
    """
    def __init__(self, token_sequences, seq_len):
        self.seq_len = seq_len
        self.data = []
        
        # Cut long token lists into fixed chunks
        for seq in token_sequences:
            for i in range(0, len(seq) - seq_len, seq_len):
                self.data.append(seq[i:i + seq_len + 1]) # +1 to allow shifted labels

        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Transformer input is tokens
        # Transformer target is shifted tokens
        x = seq[:-1]
        y = seq[1:]
        return x, y