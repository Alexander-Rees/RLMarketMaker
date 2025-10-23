"""SB3-compatible wrapper for MarketMakerEnv."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class SB3MarketMakerWrapper(gym.Wrapper):
    """Wrapper to make MarketMakerEnv compatible with SB3."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        
    def step(self, action):
        """Step the environment and track episode info."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode info for SB3
        self.episode_reward += reward
        self.episode_length += 1
        
        # Add episode info to info dict for SB3
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
            # Reset for next episode
            self.episode_reward = 0.0
            self.episode_length = 0
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        return obs, info
