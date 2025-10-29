#!/usr/bin/env python3
"""Minimal training script for RL Market Maker."""

import argparse
import os
import sys
import random
import numpy as np
import torch
import pandas as pd
import yaml
import hashlib
from pathlib import Path
from typing import Dict, Any
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO, RolloutBuffer
from rlmarketmaker.utils.io import write_json, append_csv_row, ensure_dir


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
    print(f"Loaded config: {config}")
    
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    
    # Create feed based on config
    env_config = config.get('env', {})
    feed_type = env_config.get('feed_type', 'SyntheticFeed')
    print(f"Using feed type: {feed_type}")
    if feed_type == 'PolygonReplayFeed':
        from rlmarketmaker.data.feeds import PolygonReplayFeed
        feed = PolygonReplayFeed(seed=seed)
        print("Created PolygonReplayFeed for historical data training")
    else:
        feed = SyntheticFeed(seed=seed)
        print("Created SyntheticFeed for synthetic data training")
    
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    env = RecordEpisodeStats(env)
    
    # Get environment dimensions
    state_dim = env.env.observation_space.shape[0]
    action_dims = env.env.action_space.nvec.tolist()
    
    print(f"State dimension: {state_dim}, Action dimensions: {action_dims}")
    
    # Create PPO trainer
    ppo_config = config.get('ppo', {})
    trainer = MinPPO(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=ppo_config.get('learning_rate', 0.0003),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5)
    )
    
    # Create rollout buffer
    buffer_size = ppo_config.get('n_steps', 2048)
    buffer = RolloutBuffer(buffer_size, state_dim, action_dims)
    
    # Training parameters
    total_timesteps = ppo_config.get('total_timesteps', 100000)
    n_epochs = ppo_config.get('n_epochs', 4)
    batch_size = ppo_config.get('batch_size', 64)
    save_freq = ppo_config.get('save_freq', 10000)
    
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
    
    # Compute final metrics
    final_metrics = {}
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        # Compute aggregate metrics
        if len(df) > 0:
            final_metrics = {
                'mean_pnl': float(df['pnl'].mean()) if 'pnl' in df.columns else 0.0,
                'mean_sharpe': float(df['sharpe'].mean()) if 'sharpe' in df.columns else 0.0,
                'mean_fill_rate': float(df['fill_rate'].mean()) if 'fill_rate' in df.columns else 0.0,
                'mean_inv_var': float(df['inv_var'].mean()) if 'inv_var' in df.columns else 0.0,
                'mean_max_dd': float(df['max_dd'].mean()) if 'max_dd' in df.columns else 0.0,
                'total_timesteps': total_timesteps,
                'seed': seed,
                'config_path': str(config_path),
                'checkpoint_path': str(final_checkpoint)
            }
            
            # Compute config hash for reproducibility
            try:
                with open(config_path, 'r') as f:
                    config_str = f.read()
                    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
                    final_metrics['config_hash'] = config_hash
            except:
                pass
    
    # Save metrics.json
    if final_metrics:
        metrics_path = checkpoint_dir / 'metrics.json'
        write_json(str(metrics_path), final_metrics)
    
    # Save config snapshot
    try:
        config_snapshot_path = checkpoint_dir / 'run_config.yaml'
        ensure_dir(str(config_snapshot_path))
        with open(config_snapshot_path, 'w') as f:
            yaml.dump(config, f)
    except Exception as e:
        print(f"Warning: Could not save config snapshot: {e}")
    
    print(f"Training completed!")
    print(f"Final model saved to: {final_checkpoint}")
    if csv_data:
        print(f"Metrics saved to: {csv_path}")
    if final_metrics:
        print(f"Final metrics saved to: {checkpoint_dir / 'metrics.json'}")
    
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
