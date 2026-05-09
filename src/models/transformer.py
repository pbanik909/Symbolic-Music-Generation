import torch
import torch.nn as nn
import math

class MusicTransformer(nn.Module):
    """
    Task 3: Transformer-based autoregressive model for music generation.
    Uses standard transformer encoder-decoder architecture to generate
    sequences of discrete REMI tokens.
    """
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, 
                    d_ff=1024, max_seq_len=2048, dropout=0.1):
        super(MusicTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token indices
            mask: Optional attention mask
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, seed_tokens, max_length=512, temperature=1.0, top_k=50, device='cpu'):
        """
        Autoregressively generate a sequence of tokens.
        
        Args:
            seed_tokens: Initial tokens (batch_size, init_len)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more diverse)
            top_k: Keep only top_k candidates for sampling
            device: Device to generate on
            
        Returns:
            generated: Generated token sequence
        """
        self.eval()
        with torch.no_grad():
            current_seq = seed_tokens.clone()
            
            for _ in range(max_length - seed_tokens.size(1)):
                # Forward pass on current sequence
                logits = self.forward(current_seq)
                
                # Get logits for next token (last position)
                next_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k, dim=-1)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # Safety check: limit sequence length
                if current_seq.size(1) >= self.max_seq_len:
                    break
            
            return current_seq


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers."""
    
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
