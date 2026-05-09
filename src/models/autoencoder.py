import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    Task 1: An unsupervised LSTM-based Autoencoder for compressing and 
    reconstructing 128-step piano roll sequences.
    """
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # --- ENCODER ---
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER ---
        self.decoder_linear = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Compresses the input sequence into a fixed-size latent vector."""
        # x shape: (batch_size, seq_len, 88)
        _, (h_n, _) = self.encoder_lstm(x)
        
        # Take the last hidden state of the top LSTM layer
        h_n_last = h_n[-1] 
        latent = self.encoder_linear(h_n_last)
        return latent

    def decode(self, latent, seq_len):
        """Autoregressively reconstructs the sequence from the latent vector."""
        batch_size = latent.size(0)
        
        # Initialize decoder hidden state from the latent vector
        h_0 = self.decoder_linear(latent).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        
        outputs = []
        # Start decoding with an empty frame (all zeros)
        current_input = torch.zeros(batch_size, 1, self.input_dim, device=latent.device)
        
        h, c = h_0, c_0
        for t in range(seq_len):
            out, (h, c) = self.decoder_lstm(current_input, (h, c))
            logits = self.output_layer(out)
            outputs.append(logits)
            
            # Autoregressive step: pass logits through sigmoid to use as next input
            current_input = torch.sigmoid(logits)
            
        # Concatenate all time steps
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        """End-to-end forward pass returning raw logits."""
        seq_len = x.size(1)
        latent = self.encode(x)
        # Returns RAW LOGITS to be consumed by BCEWithLogits / Focal Loss
        reconstructed_logits = self.decode(latent, seq_len)
        return reconstructed_logits