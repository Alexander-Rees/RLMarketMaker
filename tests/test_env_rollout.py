"""Test environment rollout functionality."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config


def test_environment_rollout():
    """Test 200-step rollout collecting returns."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    obs, _ = env.reset()
    rewards = []
    done = False
    step = 0
    
    while not done and step < 200:
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record reward
        rewards.append(reward)
        
        done = terminated or truncated
        step += 1
    
    # Assertions
    assert len(rewards) > 0, "No rewards collected"
    assert all(np.isfinite(r) for r in rewards), "Non-finite rewards found"
    assert step <= 200, "Too many steps"
    
    # Check that we can compute basic statistics
    total_reward = sum(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    assert np.isfinite(total_reward), "Total reward is not finite"
    assert np.isfinite(mean_reward), "Mean reward is not finite"
    assert np.isfinite(std_reward), "Std reward is not finite"
    
    print(f"Rollout completed: {step} steps, total reward: {total_reward:.2f}")


def test_environment_reset():
    """Test environment reset functionality."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    # First reset
    obs1, info1 = env.reset()
    assert obs1 is not None
    assert info1 is not None
    
    # Take a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Second reset
    obs2, info2 = env.reset()
    assert obs2 is not None
    assert info2 is not None
    
    # Observations should be different (due to random seed)
    assert not np.array_equal(obs1, obs2), "Reset should produce different observations"


def test_environment_action_space():
    """Test environment action space."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    # Test action space
    assert hasattr(env, 'action_space'), "Environment should have action_space"
    assert hasattr(env, 'observation_space'), "Environment should have observation_space"
    
    # Test action sampling
    action = env.action_space.sample()
    assert isinstance(action, np.ndarray), "Action should be numpy array"
    assert action.shape == (3,), "Action should have shape (3,)"
    assert all(0 <= a < n for a, n in zip(action, env.action_space.nvec)), "Action should be valid"
    
    # Test observation space
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == env.observation_space.shape, "Observation shape should match space"


def test_environment_info_dict():
    """Test environment info dictionary."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    obs, info = env.reset()
    
    # Test info dict
    assert isinstance(info, dict), "Info should be a dictionary"
    
    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Test info dict after step
    assert isinstance(info, dict), "Info should be a dictionary after step"
    
    # Test that info is JSON serializable
    import json
    try:
        json.dumps(info)
    except TypeError:
        pytest.fail("Info dict should be JSON serializable")


def test_environment_termination():
    """Test environment termination conditions."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    obs, _ = env.reset()
    step = 0
    max_steps = 1000
    
    while step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check termination flags
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        
        if terminated or truncated:
            break
        
        step += 1
    
    # Should eventually terminate
    assert step < max_steps, "Environment should terminate within max_steps"


def test_environment_reward_range():
    """Test that rewards are within reasonable range."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    obs, _ = env.reset()
    rewards = []
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Check reward properties
    assert len(rewards) > 0, "Should have collected some rewards"
    assert all(np.isfinite(r) for r in rewards), "All rewards should be finite"
    
    # Rewards should be within reasonable range (not too extreme)
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    assert min_reward > -10000, "Rewards should not be extremely negative"
    assert max_reward < 10000, "Rewards should not be extremely positive"
    
    print(f"Reward range: [{min_reward:.2f}, {max_reward:.2f}]")
