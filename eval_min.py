#!/usr/bin/env python3
"""Minimal evaluation script for RL Market Maker."""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO


def evaluate_model(checkpoint_path: str, config_path: str, n_episodes: int = 10, seed: int = 42):
    """Evaluate trained model."""
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    print("Creating environment...")
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    
    print(f"State dimension: {state_dim}, Action dimensions: {action_dims}")
    
    # Create trainer and load model
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
    
    # Load checkpoint
    trainer.load(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    episode_pnls = []
    episode_inventories = []
    episode_fill_rates = []
    
    print(f"Evaluating for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_inventory = []
        episode_fills = 0
        episode_orders = 0
        done = False
        
        while not done:
            # Normalize observation
            norm_obs = trainer.vec_normalize.normalize(obs.reshape(1, -1)).squeeze()
            
            # Get action (deterministic)
            action, _ = trainer.get_action(norm_obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record stats
            episode_reward += reward
            episode_length += 1
            episode_inventory.append(info.get('inventory', 0))
            
            if reward != 0:  # Fill occurred
                episode_fills += 1
            episode_orders += 1
            
            done = terminated or truncated
            
            if episode_length > 1000:  # Safety limit
                break
        
        # Calculate episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_pnls.append(info.get('total_pnl', episode_reward))
        episode_inventories.append(np.mean(episode_inventory))
        episode_fill_rates.append(episode_fills / episode_orders if episode_orders > 0 else 0)
        
        print(f"Episode {episode + 1}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"PnL={info.get('total_pnl', episode_reward):.2f}, "
              f"Fill Rate={episode_fills/episode_orders if episode_orders > 0 else 0:.2f}")
    
    # Calculate summary metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_pnl = np.mean(episode_pnls)
    std_pnl = np.std(episode_pnls)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    mean_fill_rate = np.mean(episode_fill_rates)
    mean_inventory = np.mean(episode_inventories)
    inv_var = np.var(episode_inventories)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Mean PnL: {mean_pnl:.2f} ± {std_pnl:.2f}")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Mean Fill Rate: {mean_fill_rate:.2f}")
    print(f"   Mean Inventory: {mean_inventory:.2f}")
    print(f"   Inventory Variance: {inv_var:.2f}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'sharpe': sharpe,
        'mean_fill_rate': mean_fill_rate,
        'mean_inventory': mean_inventory,
        'inv_var': inv_var
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate RL Market Maker with Minimal PPO')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/realistic_environment.yaml',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.checkpoint + '.pt'):
        print(f"Checkpoint file not found: {args.checkpoint}.pt")
        return
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    
    # Evaluate model
    results = evaluate_model(args.checkpoint, args.config, args.episodes, args.seed)
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
