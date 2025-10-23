"""Smoke test for PPO training."""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config


def create_env_factory(config, seed=None):
    """Create environment factory for testing."""
    def _init():
        feed = SyntheticFeed(seed=seed)
        env = RealisticMarketMakerEnv(feed, config, seed=seed)
        return env
    return _init


def test_ppo_training_smoke():
    """Test that PPO training runs without errors."""
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Load config
        config = load_config('configs/realistic_environment.yaml')
        
        # Override config for quick test
        config['total_timesteps'] = 1000  # Very short training
        config['log_dir'] = str(temp_path)
        
        # Create environment
        env_factory = create_env_factory(config, seed=42)
        train_env = make_vec_env(env_factory, n_envs=1, seed=42)
        train_env = VecMonitor(train_env)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=0.0003,
            n_steps=256,  # Small for quick test
            batch_size=32,
            n_epochs=3,
            verbose=0,
            tensorboard_log=None  # Disable tensorboard for test
        )
        
        # Train for very short time
        model.learn(total_timesteps=1000, progress_bar=False)
        
        # Check that model was trained
        assert model.num_timesteps == 1000, f"Expected 1000 timesteps, got {model.num_timesteps}"
        
        # Test prediction
        obs = train_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None, "Model should be able to predict actions"
        
        # Save model
        model_path = temp_path / 'test_model'
        model.save(str(model_path))
        
        # Check that model file exists
        assert (temp_path / 'test_model.zip').exists(), "Model file should be saved"
        
        # Save VecNormalize stats
        vecnorm_path = temp_path / 'vecnorm.pkl'
        train_env.save(str(vecnorm_path))
        
        # Check that VecNormalize file exists
        assert vecnorm_path.exists(), "VecNormalize stats should be saved"


def test_ppo_training_with_eval():
    """Test PPO training with evaluation callback."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Load config
        config = load_config('configs/realistic_environment.yaml')
        config['total_timesteps'] = 1000
        config['log_dir'] = str(temp_path)
        
        # Create training environment
        train_env_factory = create_env_factory(config, seed=42)
        train_env = make_vec_env(train_env_factory, n_envs=1, seed=42)
        train_env = VecMonitor(train_env)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # Create evaluation environment
        eval_env_factory = create_env_factory(config, seed=43)
        eval_env = make_vec_env(eval_env_factory, n_envs=1, seed=43)
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=0.0003,
            n_steps=256,
            batch_size=32,
            n_epochs=3,
            verbose=0,
            tensorboard_log=None
        )
        
        # Train with evaluation
        from stable_baselines3.common.callbacks import EvalCallback
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(temp_path / 'best_model'),
            log_path=str(temp_path / 'eval_logs'),
            eval_freq=500,
            deterministic=True,
            render=False,
            verbose=0
        )
        
        model.learn(total_timesteps=1000, callback=eval_callback, progress_bar=False)
        
        # Check that training completed
        assert model.num_timesteps == 1000
        
        # Check that evaluation files were created
        assert (temp_path / 'eval_logs').exists(), "Evaluation logs should be created"


def test_model_loading():
    """Test that trained model can be loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Load config
        config = load_config('configs/realistic_environment.yaml')
        config['total_timesteps'] = 500
        config['log_dir'] = str(temp_path)
        
        # Create environment
        env_factory = create_env_factory(config, seed=42)
        train_env = make_vec_env(env_factory, n_envs=1, seed=42)
        train_env = VecMonitor(train_env)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # Create and train model
        model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=None)
        model.learn(total_timesteps=500, progress_bar=False)
        
        # Save model
        model_path = temp_path / 'test_model'
        model.save(str(model_path))
        
        # Load model
        loaded_model = PPO.load(str(model_path))
        
        # Test that loaded model works
        obs = train_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action is not None, "Loaded model should be able to predict actions"
