#!/usr/bin/env python3
"""Trace evaluation script for replay environment."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rlmarketmaker.env.replay_market_env import ReplayMarketMakerEnv
from rlmarketmaker.data.feeds import PolygonReplayFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO


def trace_ppo_replay(checkpoint_path: str, config_path: str, steps: int, seed: int):
    """Trace PPO agent behavior on replay data."""
    print(f"Tracing PPO agent on replay data from {checkpoint_path}...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    feed = PolygonReplayFeed(seed=seed)
    env = ReplayMarketMakerEnv(feed, config['env'], seed=seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    
    # Create trainer and load model
    trainer = MinPPO(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=config['ppo'].get('learning_rate', 0.0003),
        gamma=config['ppo'].get('gamma', 0.99),
        gae_lambda=config['ppo'].get('gae_lambda', 0.95),
        clip_range=config['ppo'].get('clip_range', 0.2),
        vf_coef=config['ppo'].get('vf_coef', 0.5),
        ent_coef=config['ppo'].get('ent_coef', 0.01),
        max_grad_norm=config['ppo'].get('max_grad_norm', 0.5)
    )
    
    # Load checkpoint
    if checkpoint_path.endswith('.pt'):
        checkpoint_path = checkpoint_path[:-3]
    trainer.load(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    
    # Trace episode
    obs, _ = env.reset()
    trace_data = []
    
    for step in range(steps):
        # Get action (deterministic)
        result = trainer.get_action(obs, deterministic=True)
        if len(result) == 2:
            action, _ = result
        else:
            action, _, _ = result
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract trace data
        trace_row = {
            "t": step,
            "ts": step,
            "mid": info.get('midprice', 0),
            "bid_quote": info.get('bid_quote', 0),
            "ask_quote": info.get('ask_quote', 0),
            "filled_bid": int(info.get('filled_bid', 0)),
            "filled_ask": int(info.get('filled_ask', 0)),
            "inventory": info.get('inventory', 0),
            "cum_pnl": info.get('cumulative_pnl', 0),
            "spread": info.get('ask_quote', 0) - info.get('bid_quote', 0),
            "vol": info.get('volatility', 0),
            "action_bid_off": action[0],
            "action_ask_off": action[1],
            "reward": reward
        }
        
        trace_data.append(trace_row)
        
        obs = next_obs
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Save trace
    df = pd.DataFrame(trace_data)
    output_path = Path("artifacts/traces/PPO_Replay_trace.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved trace to {output_path}")
    
    # Print summary
    print(f"\nTrace Summary:")
    print(f"Total steps: {len(df)}")
    print(f"Total fills: {(df['filled_bid'] + df['filled_ask']).sum()}")
    print(f"Fill rate: {(df['filled_bid'] + df['filled_ask']).mean():.3f}")
    print(f"Final PnL: {df['cum_pnl'].iloc[-1]:.2f}")
    print(f"Final inventory: {df['inventory'].iloc[-1]:.2f}")
    print(f"Mean reward: {df['reward'].mean():.3f}")
    print(f"Reward std: {df['reward'].std():.3f}")
    
    return df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Trace PPO agent on replay data')
    parser.add_argument('--ckpt', type=str, default='logs/checkpoints/policy.pt',
                       help='Checkpoint path for PPO agent')
    parser.add_argument('--config', type=str, default='configs/polygon_replay.yaml',
                       help='Configuration file')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to trace')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    
    args = parser.parse_args()
    
    trace_ppo_replay(args.ckpt, args.config, args.steps, args.seed)
    print("Replay trace evaluation completed!")


if __name__ == '__main__':
    main()
