#!/usr/bin/env python3
"""Risk parameter sweep script for PPO optimization."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import itertools
import time

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import PolygonReplayFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO, RolloutBuffer


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        
    # Inventory metrics
    inv_var = np.var(episode_inventories) if episode_inventories else 0.0
    
    # Fill rate
    fill_rate = np.mean(episode_fill_rates) if episode_fill_rates else 0.0
    
    # Max Drawdown
    max_dd = np.max(episode_max_drawdowns) if episode_max_drawdowns else 0.0
    
    return {
        'total_pnl': total_pnl,
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'sharpe': sharpe,
        'inv_var': inv_var,
        'fill_rate': fill_rate,
        'max_dd': max_dd
    }


def train_sweep_run(lambda_val: float, H_val: int, seed: int, config: Dict[str, Any], 
                   timesteps: int = 500000) -> Dict[str, float]:
    """Train PPO with specific lambda and H values."""
    print(f"\n🎯 Training: λ={lambda_val}, H={H_val}, seed={seed}")
    
    # Set seeds
    set_seeds(seed)
    
    # Update config with sweep parameters
    config['lambda_inventory'] = lambda_val
    config['position_limit_threshold'] = H_val
    
    # Create directories
    log_dir = Path(config.get('log_dir', 'logs'))
    checkpoint_dir = log_dir / 'checkpoints' / f"{lambda_val}_{H_val}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    feed = PolygonReplayFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    env = RecordEpisodeStats(env)
    
    # Get environment dimensions
    state_dim = env.env.observation_space.shape[0]
    action_dims = env.env.action_space.nvec.tolist()
    
    # Create PPO trainer with learning rate decay
    ppo_config = config.get('ppo', {})
    initial_lr = ppo_config.get('learning_rate', 0.0003)
    final_lr = ppo_config.get('learning_rate_decay', 0.0001)
    
    trainer = MinPPO(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=initial_lr,
        gamma=ppo_config.get('gamma', 0.999),
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
    n_epochs = ppo_config.get('n_epochs', 4)
    batch_size = ppo_config.get('batch_size', 64)
    save_freq = ppo_config.get('save_freq', 100000)
    target_kl = ppo_config.get('target_kl', 0.02)
    
    # Create CSV logger
    csv_log_path = checkpoint_dir / f"run_seed_{seed}.csv"
    log_data = []
    
    # Training loop
    obs, _ = env.reset()
    current_timesteps = 0
    
    while current_timesteps < timesteps:
        # Collect experience
        for _ in range(buffer_size):
            action, log_prob, value = trainer.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store in buffer
            buffer.add(obs, action, reward, terminated, log_prob, value)
            
            obs = next_obs
            current_timesteps += 1
            
            if terminated or truncated:
                obs, _ = env.reset()
                
                # Log episode stats
                episode_metrics = compute_metrics(
                    env.episode_pnls, env.episode_inventories, 
                    env.episode_fill_rates, env.episode_max_drawdowns
                )
                log_data.append({
                    'timesteps': current_timesteps,
                    'episode': len(env.episode_pnls),
                    'pnl': episode_metrics.get('mean_pnl', 0),
                    'sharpe': episode_metrics.get('sharpe', 0),
                    'inv_var': episode_metrics.get('inv_var', 0),
                    'fill_rate': episode_metrics.get('fill_rate', 0),
                    'max_dd': episode_metrics.get('max_dd', 0)
                })
                
                # Clear episode stats for next episode
                env.episode_pnls = []
                env.episode_inventories = []
                env.episode_fill_rates = []
                env.episode_max_drawdowns = []
        
        # Compute advantages and returns
        last_value = trainer.get_value(obs)
        buffer.compute_gae(last_value, trainer.gamma, trainer.gae_lambda)
        
        # Update policy and value network with KL early stopping
        update_results = trainer.update(buffer, n_epochs, batch_size)
        actor_loss = update_results['actor_loss']
        value_loss = update_results['value_loss']
        entropy_loss = update_results['entropy_loss']
        kl_div = update_results['kl_div']
        
        # KL early stopping
        if kl_div > target_kl:
            print(f"Early stopping due to KL divergence: {kl_div:.4f} > {target_kl}")
            break
        
        # Learning rate decay
        progress = current_timesteps / timesteps
        current_lr = initial_lr + (final_lr - initial_lr) * progress
        trainer.optimizer.param_groups[0]['lr'] = current_lr
        
        # Entropy coefficient decay
        initial_ent_coef = ppo_config.get('ent_coef', 0.01)
        final_ent_coef = ppo_config.get('ent_coef_decay', 0.005)
        current_ent_coef = initial_ent_coef + (final_ent_coef - initial_ent_coef) * progress
        trainer.ent_coef = current_ent_coef
        
        # Clear buffer
        buffer.clear()
        
        # Save model
        if current_timesteps % save_freq == 0:
            trainer.save(str(checkpoint_dir / f"policy_seed_{seed}"))
            print(f"Model saved at {current_timesteps} timesteps.")
    
        # Final save
        trainer.save(str(checkpoint_dir / f"policy_seed_{seed}"))
    
    # Save metrics to CSV
    pd.DataFrame(log_data).to_csv(csv_log_path, index=False)
    
    # Return final metrics
    final_metrics = compute_metrics(
        env.episode_pnls, env.episode_inventories, 
        env.episode_fill_rates, env.episode_max_drawdowns
    )
    
    return {
        'lambda': lambda_val,
        'H': H_val,
        'seed': seed,
        'pnl': final_metrics.get('mean_pnl', 0),
        'sharpe': final_metrics.get('sharpe', 0),
        'inv_var': final_metrics.get('inv_var', 0),
        'fill_rate': final_metrics.get('fill_rate', 0),
        'max_dd': final_metrics.get('max_dd', 0),
        'timesteps': current_timesteps
    }


def run_risk_sweep(config_path: str, timesteps: int = 500000):
    """Run risk parameter sweep."""
    # Load base configuration
    config = load_config(config_path)
    
    # Define sweep parameters
    lambda_values = [0.0025, 0.0030, 0.0035]
    H_values = [25, 30, 40]
    seeds = [11, 23]
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts/sweeps")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results
    results = []
    
    # Run sweep
    total_runs = len(lambda_values) * len(H_values) * len(seeds)
    current_run = 0
    
    for lambda_val, H_val in itertools.product(lambda_values, H_values):
        for seed in seeds:
            current_run += 1
            print(f"\n🚀 Run {current_run}/{total_runs}")
            
            try:
                result = train_sweep_run(lambda_val, H_val, seed, config.copy(), timesteps)
                results.append(result)
                
                print(f"✅ Completed: λ={lambda_val}, H={H_val}, seed={seed}")
                print(f"   PnL: {result['pnl']:.2f}, Sharpe: {result['sharpe']:.2f}, Inv Var: {result['inv_var']:.2f}")
                
            except Exception as e:
                print(f"❌ Failed: λ={lambda_val}, H={H_val}, seed={seed}")
                print(f"   Error: {e}")
                results.append({
                    'lambda': lambda_val,
                    'H': H_val,
                    'seed': seed,
                    'pnl': 0.0,
                    'sharpe': 0.0,
                    'inv_var': 0.0,
                    'fill_rate': 0.0,
                    'max_dd': 0.0,
                    'timesteps': 0,
                    'error': str(e)
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = artifacts_dir / "risk_sweep.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n📊 Sweep Results Summary:")
    print(f"   Total runs: {len(results)}")
    print(f"   Results saved to: {results_path}")
    
    # Find best runs
    successful_results = results_df[results_df['pnl'] > 0]
    if len(successful_results) > 0:
        best_pnl = successful_results.loc[successful_results['pnl'].idxmax()]
        best_sharpe = successful_results.loc[successful_results['sharpe'].idxmax()]
        
        print(f"\n🏆 Best Results:")
        print(f"   Best PnL: λ={best_pnl['lambda']}, H={best_pnl['H']}, seed={best_pnl['seed']}")
        print(f"     PnL: {best_pnl['pnl']:.2f}, Sharpe: {best_pnl['sharpe']:.2f}")
        print(f"   Best Sharpe: λ={best_sharpe['lambda']}, H={best_sharpe['H']}, seed={best_sharpe['seed']}")
        print(f"     PnL: {best_sharpe['pnl']:.2f}, Sharpe: {best_sharpe['sharpe']:.2f}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk parameter sweep for PPO optimization.")
    parser.add_argument("--config", type=str, default="configs/ppo_improved.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Number of training timesteps per run.")
    
    args = parser.parse_args()
    
    run_risk_sweep(args.config, args.timesteps)
