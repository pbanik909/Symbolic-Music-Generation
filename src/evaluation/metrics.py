import numpy as np

def calculate_reward(tokens, tokenizer):
    """
    A heuristic reward function acting as a proxy for Human Feedback.
    Rewards musical structure and penalizes noise/repetition.
    """
    if len(tokens) < 10:
        return -1.0 # Penalize extremely short sequences

    # Convert token IDs back to strings to analyze them
    try:
        # miditok v3 representation
        token_strings = [tokenizer[t] for t in tokens]
    except Exception:
        return 0.0

    reward = 0.0
    
    # 1. Penalize repetitive token loops (e.g., Pitch_60, Pitch_60, Pitch_60)
    repeats = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            repeats += 1
    
    repeat_ratio = repeats / len(tokens)
    if repeat_ratio > 0.1:
        reward -= 2.0  # Heavy penalty for getting stuck in a loop
        
    # 2. Reward Pitch Variety
    pitch_tokens = [t for t in token_strings if 'Pitch' in t]
    unique_pitches = len(set(pitch_tokens))
    if unique_pitches > 5 and unique_pitches < 25:
        reward += 1.0  # Good melodic variety
    elif unique_pitches <= 5:
        reward -= 1.0  # Too boring

    # 3. Reward valid structural grammar (TimeShift -> Pitch -> Velocity -> Duration)
    # Just checking for presence of these tokens as a basic sanity check
    has_time = any('TimeShift' in t or 'TimeSig' in t for t in token_strings)
    has_dur = any('Duration' in t for t in token_strings)
    if has_time and has_dur:
        reward += 0.5
        
    return reward