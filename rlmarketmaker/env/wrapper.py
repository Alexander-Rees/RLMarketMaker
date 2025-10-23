"""Environment wrapper to fix SB3 compatibility issues."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class MarketMakerWrapper(gym.Wrapper):
    """Wrapper to fix SB3 compatibility issues."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def step(self, action):
        """Step the environment and track episode info."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode info for SB3
        if hasattr(self, 'episode_reward'):
            self.episode_reward += reward
            self.episode_length += 1
        else:
            self.episode_reward = reward
            self.episode_length = 1
        
        # Store episode info when episode ends
        if terminated or truncated:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            self.episode_count += 1
            
            # Reset for next episode
            self.episode_reward = 0
            self.episode_length = 0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        return obs, info
