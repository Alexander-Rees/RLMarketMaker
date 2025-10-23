#!/usr/bin/env python3
"""Minimal training script for RL Market Maker."""

import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import time

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO, RolloutBuffer


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed}")


class RecordEpisodeStats:
    """Lightweight episode statistics recorder."""
    
    def __init__(self, env):
        self.env = env
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_pnls = []
        self.episode_inventories = []
        self.episode_fill_rates = []
        self.episode_max_drawdowns = []
        
    def step(self, action):
        """Step environment and record stats."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record episode stats on terminal
        if terminated or truncated:
            if hasattr(self.env, 'episode_stats'):
                stats = self.env.episode_stats
                self.episode_rewards.append(stats.get('episode_reward', 0))
                self.episode_lengths.append(stats.get('episode_length', 0))
                self.episode_pnls.append(stats.get('episode_pnl', 0))
                self.episode_inventories.append(stats.get('mean_inventory', 0))
                self.episode_fill_rates.append(stats.get('fill_rate', 0))
                self.episode_max_drawdowns.append(stats.get('max_drawdown', 0))
        
        return obs, reward, terminated, truncated, info
    
    def reset(self):
        """Reset environment."""
        return self.env.reset()


def compute_metrics(episode_pnls, episode_inventories, episode_fill_rates, episode_max_drawdowns):
    """Compute performance metrics."""
    if not episode_pnls:
        return {}
    
    # PnL metrics
    total_pnl = sum(episode_pnls)
    mean_pnl = np.mean(episode_pnls)
    std_pnl = np.std(episode_pnls)
    
    # Sharpe ratio
    if std_pnl > 0:
        sharpe = mean_pnl / std_pnl
    else:
        sharpe = 0.0
    
    # Inventory variance
    if episode_inventories:
        inv_var = np.var(episode_inventories)
    else:
        inv_var = 0.0
    
    # Fill rate
    if episode_fill_rates:
        fill_rate = np.mean(episode_fill_rates)
    else:
        fill_rate = 0.0
    
    # Max drawdown
    if episode_max_drawdowns:
        max_dd = np.max(episode_max_drawdowns)
    else:
        max_dd = 0.0
    
    return {
        'total_pnl': total_pnl,
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'sharpe': sharpe,
        'inv_var': inv_var,
        'fill_rate': fill_rate,
        'max_dd': max_dd
    }


def train_minimal_ppo(config_path: str, seed: int = 42):
    """Train using minimal PPO implementation."""
    # Load configuration
    config = load_config(config_path)
    
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    env = RecordEpisodeStats(env)
    
    # Get environment dimensions
    state_dim = env.env.observation_space.shape[0]
    action_dims = env.env.action_space.nvec.tolist()
    
    print(f"State dimension: {state_dim}, Action dimensions: {action_dims}")
    
    # Create PPO trainer
    trainer = MinPPO(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=config.get('learning_rate', 0.0003),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        vf_coef=config.get('vf_coef', 0.5),
        ent_coef=config.get('ent_coef', 0.01),
        max_grad_norm=config.get('max_grad_norm', 0.5)
    )
    
    # Create rollout buffer
    buffer_size = config.get('n_steps', 2048)
    buffer = RolloutBuffer(buffer_size, state_dim, action_dims)
    
    # Training parameters
    total_timesteps = config.get('total_timesteps', 100000)
    n_epochs = config.get('n_epochs', 4)
    batch_size = config.get('batch_size', 64)
    save_freq = config.get('save_freq', 10000)
    
    print(f"Starting minimal PPO training for {total_timesteps} timesteps...")
    print(f"Model will be saved to {checkpoint_dir}")
    
    # Create CSV logger
    timestamp = int(time.time())
    csv_path = log_dir / f'run_{timestamp}.csv'
    csv_data = []
    
    # Training loop
    timesteps = 0
    episode = 0
    
    obs, _ = env.reset()
    
    while timesteps < total_timesteps:
        # Collect rollout
        for step in range(buffer_size):
            # Normalize observation
            norm_obs = trainer.vec_normalize.normalize(obs.reshape(1, -1)).squeeze()
            
            # Get action
            action, log_prob, value = trainer.get_action(norm_obs)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            buffer.add(norm_obs, action, reward, terminated or truncated, log_prob, value)
            
            obs = next_obs
            timesteps += 1
            
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset()
                
                # Log episode metrics
                if len(env.episode_rewards) > 0:
                    metrics = compute_metrics(
                        env.episode_pnls[-1:],
                        env.episode_inventories[-1:],
                        env.episode_fill_rates[-1:],
                        env.episode_max_drawdowns[-1:]
                    )
                    
                    csv_data.append({
                        'episode': episode,
                        'timesteps': timesteps,
                        'pnl': metrics['total_pnl'],
                        'sharpe': metrics['sharpe'],
                        'inv_var': metrics['inv_var'],
                        'fill_rate': metrics['fill_rate'],
                        'max_dd': metrics['max_dd']
                    })
                    
                    print(f"Episode {episode}: PnL={metrics['total_pnl']:.2f}, "
                          f"Sharpe={metrics['sharpe']:.2f}, "
                          f"Fill Rate={metrics['fill_rate']:.2f}")
        
        # Update policy
        if buffer.size > 0:
            losses = trainer.update(buffer, n_epochs, batch_size)
            print(f"Update losses: Actor={losses['actor_loss']:.4f}, "
                  f"Value={losses['value_loss']:.4f}, "
                  f"Entropy={losses['entropy_loss']:.4f}")
        
        # Update normalization
        if buffer.size > 0:
            trainer.vec_normalize.update(buffer.states[:buffer.size])
        
        # Clear buffer
        buffer.clear()
        
        # Save checkpoint
        if timesteps % save_freq == 0:
            checkpoint_path = checkpoint_dir / f'policy_step_{timesteps}'
            trainer.save(str(checkpoint_path))
            print(f"Checkpoint saved at step {timesteps}")
    
    # Save final model
    final_checkpoint = checkpoint_dir / 'policy'
    trainer.save(str(final_checkpoint))
    
    # Save CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    
    print(f"Training completed!")
    print(f"Final model saved to: {final_checkpoint}")
    print(f"Metrics saved to: {csv_path}")
    
    return trainer, final_checkpoint


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RL Market Maker with Minimal PPO')
    parser.add_argument('--config', type=str, default='configs/realistic_environment.yaml',
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
    trainer, checkpoint_path = train_minimal_ppo(args.config, args.seed)
    
    print("Minimal PPO training completed successfully!")


if __name__ == '__main__':
    main()
