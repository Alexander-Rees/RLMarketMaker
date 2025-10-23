"""Smoke test for minimal trainer."""

import pytest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO, RolloutBuffer


def test_minimal_ppo_creation():
    """Test that minimal PPO can be created."""
    config = load_config('configs/realistic_environment.yaml')
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    
    trainer = MinPPO(state_dim, action_dims)
    assert trainer is not None
    assert trainer.policy is not None
    assert trainer.optimizer is not None


def test_minimal_ppo_training_smoke():
    """Test that minimal PPO training runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Load config
        config = load_config('configs/realistic_environment.yaml')
        config['total_timesteps'] = 1000  # Very short training
        config['log_dir'] = str(temp_path)
        
        # Create environment
        feed = SyntheticFeed(seed=42)
        env = RealisticMarketMakerEnv(feed, config, seed=42)
        
        # Get dimensions
        state_dim = env.observation_space.shape[0]
        action_dims = env.action_space.nvec.tolist()
        
        # Create trainer
        trainer = MinPPO(state_dim, action_dims)
        
        # Create buffer
        buffer = RolloutBuffer(100, state_dim, action_dims)
        
        # Short training loop
        obs, _ = env.reset()
        for step in range(100):
            # Get action
            action, log_prob, value = trainer.get_action(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Add to buffer
            buffer.add(obs, action, reward, terminated or truncated, log_prob, value)
            
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Update policy
        if buffer.size > 0:
            losses = trainer.update(buffer, epochs=2, batch_size=32)
            assert 'actor_loss' in losses
            assert 'value_loss' in losses
            assert 'entropy_loss' in losses
            assert 'total_loss' in losses
        
        # Test saving/loading
        checkpoint_path = temp_path / 'test_model'
        trainer.save(str(checkpoint_path))
        
        # Check files exist
        assert (temp_path / 'test_model.pt').exists()
        assert (temp_path / 'test_model_vecnorm.pkl').exists()
        
        # Test loading
        new_trainer = MinPPO(state_dim, action_dims)
        new_trainer.load(str(checkpoint_path))
        
        # Test action
        action, _ = new_trainer.get_action(obs, deterministic=True)
        assert action is not None


def test_rollout_buffer():
    """Test rollout buffer functionality."""
    state_dim = 5
    action_dims = [10, 10, 5]
    buffer_size = 100
    
    buffer = RolloutBuffer(buffer_size, state_dim, action_dims)
    
    # Add some data
    for i in range(50):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(0, action_dims, size=len(action_dims)).astype(np.int32)
        reward = np.random.randn()
        done = i % 10 == 9  # Every 10th step is terminal
        log_prob = np.random.randn(len(action_dims)).astype(np.float32)
        value = np.random.randn()
        
        buffer.add(state, action, reward, done, log_prob, value)
    
    assert buffer.size == 50
    
    # Test GAE computation
    buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
    assert len(buffer.advantages) == buffer.size
    assert len(buffer.returns) == buffer.size
    
    # Test batch sampling
    batch = buffer.get_batch(10)
    assert len(batch['states']) == 10
    assert len(batch['actions']) == 10
    assert len(batch['advantages']) == 10


def test_vec_normalize():
    """Test vector normalization."""
    from rlmarketmaker.agents.min_ppo import VecNormalizeLite
    
    obs_dim = 5
    vec_norm = VecNormalizeLite(obs_dim)
    
    # Test normalization before any updates
    obs = np.random.randn(10, obs_dim).astype(np.float32)
    norm_obs = vec_norm.normalize(obs)
    np.testing.assert_array_equal(obs, norm_obs)  # Should be unchanged initially
    
    # Update statistics
    vec_norm.update(obs)
    
    # Test normalization after update
    norm_obs = vec_norm.normalize(obs)
    assert norm_obs.shape == obs.shape
    
    # Test saving/loading
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'vecnorm.pkl'
        vec_norm.save(str(temp_path))
        
        new_vec_norm = VecNormalizeLite(obs_dim)
        new_vec_norm.load(str(temp_path))
        
        # Test that loaded stats work
        norm_obs2 = new_vec_norm.normalize(obs)
        np.testing.assert_array_almost_equal(norm_obs, norm_obs2)


def test_train_min_script():
    """Test that train_min.py can be imported and run briefly."""
    # This is a basic import test
    try:
        import train_min
        assert hasattr(train_min, 'train_minimal_ppo')
        assert hasattr(train_min, 'main')
    except ImportError as e:
        pytest.fail(f"Could not import train_min: {e}")


def test_eval_min_script():
    """Test that eval_min.py can be imported."""
    try:
        import eval_min
        assert hasattr(eval_min, 'evaluate_model')
        assert hasattr(eval_min, 'main')
    except ImportError as e:
        pytest.fail(f"Could not import eval_min: {e}")
