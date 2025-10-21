#!/usr/bin/env python3
"""Training script for RL Market Maker."""

import argparse
import os
import random
import numpy as np
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from rlmarketmaker.env.market_env import MarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config, get_config_path
from rlmarketmaker.utils.logging import EpisodeLogger, TrainingLogger


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(seed)
    print(f"Set random seed to {seed}")


def create_env(config: dict, seed: int = None):
    """Create market making environment."""
    # Create data feed
    feed = SyntheticFeed(seed=seed)
    
    # Create environment
    env = MarketMakerEnv(feed, config, seed=seed)
    
    return env


def create_ppo_model(env, config: dict):
    """Create PPO model with configuration."""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        verbose=config.get('verbose', 1),
        tensorboard_log=config.get('tensorboard_log', 'logs/tensorboard')
    )
    
    return model


def create_callbacks(config: dict, eval_env, model_save_path: str):
    """Create training callbacks."""
    callbacks = []
    
    # Evaluation callback
    if config.get('eval_freq', 10000) > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_save_path,
            log_path=model_save_path,
            eval_freq=config.get('eval_freq', 10000),
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    # Checkpoint callback
    if config.get('save_freq', 50000) > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=config.get('save_freq', 50000),
            save_path=model_save_path,
            name_prefix='checkpoint'
        )
        callbacks.append(checkpoint_callback)
    
    return callbacks


def apply_curriculum_learning(model, step: int, config: dict):
    """Apply curriculum learning to gradually increase difficulty."""
    if not config.get('curriculum', False):
        return
    
    # Calculate curriculum progress
    curriculum_steps = config.get('curriculum_steps', 500000)
    progress = min(step / curriculum_steps, 1.0)
    
    # Adjust environment parameters
    initial_fee_mult = config.get('initial_fee_multiplier', 0.1)
    final_fee_mult = config.get('final_fee_multiplier', 1.0)
    
    current_fee_mult = initial_fee_mult + (final_fee_mult - initial_fee_mult) * progress
    
    # Update environment parameters
    if hasattr(model.env, 'envs'):
        for env in model.env.envs:
            if hasattr(env, 'reward_calc'):
                env.reward_calc.transaction_fee = config.get('transaction_fee', 0.0001) * current_fee_mult


def train_model(config_path: str, seed: int = 42):
    """Train PPO model on market making environment."""
    # Load configuration
    config = load_config(config_path)
    
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    model_save_path = log_dir / 'models'
    model_save_path.mkdir(exist_ok=True)
    
    # Create environments
    train_env = create_env(config, seed=seed)
    eval_env = create_env(config, seed=seed + 1)
    
    # Wrap with VecNormalize if specified
    if config.get('normalize_observations', True):
        train_env = DummyVecEnv([lambda: train_env])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=config.get('normalize_rewards', False))
        
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # Create PPO model
    model = create_ppo_model(train_env, config)
    
    # Create callbacks
    callbacks = create_callbacks(config, eval_env, str(model_save_path))
    
    # Create loggers
    episode_logger = EpisodeLogger(str(log_dir))
    training_logger = TrainingLogger(str(log_dir))
    
    # Training loop with curriculum learning
    total_timesteps = config.get('total_timesteps', 1000000)
    log_interval = config.get('log_interval', 10)
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Model will be saved to {model_save_path}")
    print(f"Logs will be saved to {log_dir}")
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = model_save_path / 'final_model'
    model.save(str(final_model_path))
    
    # Save VecNormalize stats
    if config.get('normalize_observations', True):
        vec_normalize_path = model_save_path / 'vec_normalize.pkl'
        train_env.save(str(vec_normalize_path))
    
    # Close loggers
    episode_logger.close()
    training_logger.close()
    
    print(f"Training completed! Model saved to {final_model_path}")
    print(f"Episode logs: {episode_logger.get_log_path()}")
    print(f"Training logs: {training_logger.get_log_path()}")
    
    return model, train_env, eval_env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RL Market Maker')
    parser.add_argument('--config', type=str, default='configs/ppo.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Available configs:")
        for config_file in Path('configs').glob('*.yaml'):
            print(f"  - {config_file}")
        return
    
    # Train model
    model, train_env, eval_env = train_model(args.config, args.seed)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
