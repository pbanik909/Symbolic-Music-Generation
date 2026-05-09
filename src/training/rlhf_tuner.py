import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class RLHFTuner:
    """
    Fine-tunes a pretrained Transformer using the REINFORCE algorithm.
    """
    def __init__(self, model, tokenizer, reward_fn, device, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = device
        # Use a much smaller learning rate for RL to prevent catastrophic forgetting
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def generate_episode(self, start_tokens, max_len=128):
        """Generates a sequence and keeps track of the log probabilities of chosen tokens."""
        self.model.train() # Need gradients flowing
        tokens = start_tokens.tolist()
        log_probs = []
        
        # Sequence generation length (context window)
        seq_len_max = 256
        
        for _ in range(max_len):
            x = torch.tensor([tokens[-seq_len_max:]]).to(self.device)
            logits = self.model(x)
            
            # Get logits for the next token
            next_token_logits = logits[0, -1, :]
            
            # Create a categorical distribution to sample from
            probs = F.softmax(next_token_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # Sample action (next token)
            action = dist.sample()
            
            # Save the log probability of the action we took
            log_probs.append(dist.log_prob(action))
            tokens.append(action.item())
            
        return tokens, torch.stack(log_probs)

    def train_step(self, start_tokens):
        """Runs one episode of generation, calculates reward, and updates weights."""
        self.optimizer.zero_grad()
        
        # 1. Generate sequence and get log probabilities
        tokens, log_probs = self.generate_episode(start_tokens)
        
        # 2. Calculate Reward
        reward = self.reward_fn(tokens, self.tokenizer)
        
        # 3. Policy Gradient Loss: Loss = -Reward * sum(log_prob)
        # We want to maximize reward, which means minimizing the negative expected reward
        loss = -reward * log_probs.sum()
        
        # 4. Backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return reward, loss.item()