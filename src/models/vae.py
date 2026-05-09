import torch
import torch.nn as nn

class LSTM_VAE(nn.Module):
    """
    Task 2: Variational Autoencoder for Music Generation.
    Learns a continuous probabilistic latent space to allow for smooth 
    multi-genre sampling and interpolation.
    """
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128, num_layers=2):
        super(LSTM_VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # --- ENCODER ---
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # VAE specific: predicting both mean (mu) and log variance (log_var)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER ---
        self.decoder_linear = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Passes input through encoder and returns mu and log_var."""
        _, (h_n, _) = self.encoder_lstm(x)
        h_n_last = h_n[-1] # Take last layer's hidden state
        
        mu = self.fc_mu(h_n_last)
        log_var = self.fc_log_var(h_n_last)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        The Reparameterization Trick: z = mu + std * epsilon
        Allows backpropagation through a random sampling process.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, seq_len):
        """Autoregressively reconstructs sequence from latent vector z."""
        batch_size = z.size(0)
        
        # Initialize hidden states from latent vector
        h_0 = self.decoder_linear(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        
        outputs = []
        current_input = torch.zeros(batch_size, 1, self.input_dim, device=z.device)
        
        h, c = h_0, c_0
        for t in range(seq_len):
            out, (h, c) = self.decoder_lstm(current_input, (h, c))
            logits = self.output_layer(out)
            outputs.append(logits)
            
            # Feed current prediction (as probabilities) into next time step
            current_input = torch.sigmoid(logits)
            
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        """End-to-end forward pass returning logits, mu, and log_var."""
        seq_len = x.size(1)
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_logits = self.decode(z, seq_len)
        
        return reconstructed_logits, mu, log_var