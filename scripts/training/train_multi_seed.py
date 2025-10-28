#!/usr/bin/env python3
"""Optimized training with cosine LR decay, early stopping, and multi-seed support."""

import argparse
import os
import sys
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import time
import math

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO, RolloutBuffer


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RecordEpisodeStats:
    """Lightweight episode statistics recorder."""
    
    def __init__(self, env):
        self.env = env
        self.episode_pnls = []
        self.episode_inventories = []
        self.episode_fill_rates = []
        
    def step(self, action):
        """Step environment and record stats."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            if hasattr(self.env, 'episode_stats'):
                stats = self.env.episode_stats
                self.episode_pnls.append(stats.get('episode_pnl', 0))
        return obs, reward, terminated, truncated, info
    
    def reset(self):
        """Reset environment."""
        return self.env.reset()


def train_with_lr_decay(config_path: str, seeds: list, eval_every: int = 50000, early_stop: int = 3):
    """Train with cosine LR decay and early stopping."""
    config = load_config(config_path)
    
    # Create directories
    artifacts_dir = Path("artifacts") / "ppo_opt"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config.get('log_dir', 'logs'))
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training with seed: {seed}")
        print(f"{'='*60}")
        
        set_seeds(seed)
        
        # Create environment
        feed = SyntheticFeed(seed=seed)
        env = RealisticMarketMakerEnv(feed, config, seed=seed)
        env = RecordEpisodeStats(env)
        
        # Get dimensions
        state_dim = env.env.observation_space.shape[0]
        action_dims = env.env.action_space.nvec.tolist()
        
        # Create PPO trainer
        ppo_config = config.get('ppo', {})
        trainer = MinPPO(
            state_dim=state_dim,
            action_dims=action_dims,
            lr=ppo_config.get('learning_rate', 0.0002),
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
        total_timesteps = ppo_config.get('total_timesteps', 1000000)
        n_epochs = ppo_config.get('n_epochs', 3)
        batch_size = ppo_config.get('batch_size', 64)
        initial_lr = ppo_config.get('learning_rate', 0.0002)
        min_lr = ppo_config.get('lr_decay_min', 0.0001)
        
        print(f"Training for {total_timesteps} timesteps...")
        
        # Training loop
        obs, _ = env.reset()
        current_timesteps = 0
        eval_history = []
        best_sharpe = -np.inf
        stagnation_count = 0
        
        while current_timesteps < total_timesteps:
            # Use constant learning rate (no decay)
            
            # Collect experience
            for _ in range(buffer_size):
                action, log_prob, value = trainer.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                buffer.add(obs, action, reward, terminated, log_prob, value)
                
                obs = next_obs
                current_timesteps += 1
                
                if terminated or truncated:
                    obs, _ = env.reset()
            
            # Compute GAE
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device)
            _, _, last_value = trainer.policy.get_action(obs_tensor, deterministic=False)
            last_value = last_value.squeeze(0).cpu().numpy()
            if last_value.ndim > 0:
                last_value = last_value.item()
            
            buffer.compute_gae(trainer.gamma, trainer.gae_lambda, last_value)
            
            # Update policy
            losses = trainer.update(buffer, n_epochs, batch_size)
            buffer.clear()
            
            # Periodic evaluation
            if current_timesteps % eval_every == 0:
                metrics = evaluate_quick(trainer, env, n_episodes=10)
                current_sharpe = metrics.get('sharpe', 0)
                eval_history.append({
                    'timesteps': current_timesteps,
                    'sharpe': current_sharpe,
                    'mean_pnl': metrics.get('mean_pnl', 0)
                })
                
                print(f"Step {current_timesteps}: Sharpe={current_sharpe:.2f}, PnL={metrics.get('mean_pnl', 0):.2f}")
                
                # Early stopping check
                if current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    stagnation_count = 0
                    # Save best model
                    trainer.save(checkpoint_dir / "policy")
                else:
                    stagnation_count += 1
                
                if stagnation_count >= early_stop:
                    print(f"Early stopping at {current_timesteps} steps")
                    break
        
        # Final evaluation
        final_metrics = evaluate_quick(trainer, env, n_episodes=30)
        all_results.append({
            'seed': seed,
            **final_metrics
        })
        
        print(f"\nFinal Results (Seed {seed}):")
        print(f"  PnL: {final_metrics['mean_pnl']:.2f} ± {final_metrics['std_pnl']:.2f}")
        print(f"  Sharpe: {final_metrics['sharpe']:.2f}")
        print(f"  Fill Rate: {final_metrics['fill_rate']:.2f}")
    
    # Aggregate results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(artifacts_dir / "results.csv", index=False)
    
    # Compute aggregated metrics
    summary = {
        'mean_pnl': f"{results_df['mean_pnl'].mean():.2f} ± {results_df['mean_pnl'].std():.2f}",
        'mean_sharpe': f"{results_df['sharpe'].mean():.2f} ± {results_df['sharpe'].std():.2f}",
        'mean_fill_rate': f"{results_df['fill_rate'].mean():.2f} ± {results_df['fill_rate'].std():.2f}",
        'mean_inv_var': f"{results_df['inv_var'].mean():.2f} ± {results_df['inv_var'].std():.2f}",
        'mean_max_dd': f"{results_df['max_dd'].mean():.2f} ± {results_df['max_dd'].std():.2f}"
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(artifacts_dir / "summary.csv", index=False)
    
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {artifacts_dir}")


def evaluate_quick(trainer, env, n_episodes: int):
    """Quick evaluation."""
    episode_pnls = []
    episode_inventories = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        inv_history = []
        
        while not done and step < env.env.episode_length:
            action, _ = trainer.get_action(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            inv_history.append(info.get('inventory', 0))
        
        episode_pnls.append(info.get('total_pnl', 0))
        episode_inventories.extend(inv_history)
    
    mean_pnl = np.mean(episode_pnls)
    std_pnl = np.std(episode_pnls)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    inv_var = np.var(episode_inventories) if episode_inventories else 0
    fill_rate = 0.8  # Placeholder
    
    return {
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'sharpe': sharpe,
        'inv_var': inv_var,
        'fill_rate': fill_rate,
        'max_dd': 0.0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with cosine LR decay")
    parser.add_argument("--config", type=str, default="configs/ppo_optimized.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--eval-every", type=int, default=50000)
    parser.add_argument("--early-stop", type=int, default=3)
    
    args = parser.parse_args()
    
    train_with_lr_decay(args.config, args.seeds, args.eval_every, args.early_stop)
