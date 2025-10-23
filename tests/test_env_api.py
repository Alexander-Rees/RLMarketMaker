"""Test environment API compatibility."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.env_checker import check_env
from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config


def test_environment_creation():
    """Test that environment can be created successfully."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None


def test_environment_api():
    """Test environment API compatibility with SB3."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    # Run SB3 environment checker
    check_env(env, warn=True)


def test_environment_step_types():
    """Test that step returns correct types."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    obs, info = env.reset()
    
    # Test 5 steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check types
        assert isinstance(obs, np.ndarray), f"obs should be np.ndarray, got {type(obs)}"
        assert obs.dtype == np.float32, f"obs should be float32, got {obs.dtype}"
        assert obs.shape == env.observation_space.shape, f"obs shape mismatch"
        
        assert isinstance(reward, (int, float, np.number)), f"reward should be scalar, got {type(reward)}"
        assert isinstance(terminated, bool), f"terminated should be bool, got {type(terminated)}"
        assert isinstance(truncated, bool), f"truncated should be bool, got {type(truncated)}"
        assert isinstance(info, dict), f"info should be dict, got {type(info)}"
        
        # Check info is JSON serializable
        for key, value in info.items():
            if isinstance(value, np.integer):
                info[key] = int(value)
            elif isinstance(value, np.floating):
                info[key] = float(value)
            elif isinstance(value, np.ndarray):
                info[key] = value.tolist()


def test_environment_observation_space():
    """Test observation space properties."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    obs, _ = env.reset()
    
    # Check observation space
    assert env.observation_space.contains(obs), "Observation should be within observation space"
    assert obs.shape == (5,), f"Expected obs shape (5,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"


def test_environment_action_space():
    """Test action space properties."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    # Test action space
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Action should be within action space"
    assert len(action) == 3, f"Expected 3 action components, got {len(action)}"
    assert all(isinstance(x, (int, np.integer)) for x in action), "All action components should be integers"
