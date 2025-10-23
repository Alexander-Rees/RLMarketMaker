#!/usr/bin/env python3
"""Trace evaluation script to capture per-timestep agent behavior."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO
from rlmarketmaker.agents.baselines import (
    FixedSpreadStrategy, RandomStrategy, InventoryMeanReversionStrategy, AvellanedaStoikovStrategy
)


def trace_ppo_agent(checkpoint_path: str, config_path: str, steps: int, seed: int):
    """Trace PPO agent behavior."""
    print(f"Tracing PPO agent from {checkpoint_path}...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    
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
    
    # Load checkpoint (remove .pt extension if present)
    if checkpoint_path.endswith('.pt'):
        checkpoint_path = checkpoint_path[:-3]
    trainer.load(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    
    # Trace episode
    obs, _ = env.reset()
    trace_data = []
    
    for step in range(steps):
        # Normalize observation
        norm_obs = trainer.vec_normalize.normalize(obs.reshape(1, -1)).squeeze()
        
        # Get action (deterministic)
        action, _ = trainer.get_action(norm_obs, deterministic=True)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract trace data
        trace_row = {
            "t": step,
            "ts": step,  # Use step as timestamp
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
            "action_ask_off": action[1]
        }
        
        trace_data.append(trace_row)
        
        obs = next_obs
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Save trace
    df = pd.DataFrame(trace_data)
    output_path = Path("artifacts/traces/PPO_RL_Improved_trace.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved trace to {output_path}")
    
    return df


def trace_baseline_agent(baseline_name: str, config_path: str, steps: int, seed: int):
    """Trace baseline agent behavior."""
    print(f"Tracing {baseline_name} baseline agent...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    
    # Create baseline agent
    if baseline_name.upper() == "AS":
        agent = AvellanedaStoikovStrategy()
    elif baseline_name.upper() == "FIXED":
        agent = FixedSpreadStrategy()
    elif baseline_name.upper() == "RANDOM":
        agent = RandomStrategy()
    elif baseline_name.upper() == "INV":
        agent = InventoryMeanReversionStrategy()
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    # Trace episode
    obs, _ = env.reset()
    trace_data = []
    
    for step in range(steps):
        # Get market state for agent
        market_state = {
            'midprice': env.current_tick.midprice,
            'spread': env.current_tick.spread,
            'bid': env.current_tick.midprice - env.current_tick.spread/2,
            'ask': env.current_tick.midprice + env.current_tick.spread/2
        }
        
        # Get action from agent
        action = agent.get_action(obs, market_state)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract trace data
        trace_row = {
            "t": step,
            "ts": step,  # Use step as timestamp
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
            "action_ask_off": action[1]
        }
        
        trace_data.append(trace_row)
        
        obs = next_obs
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Save trace
    df = pd.DataFrame(trace_data)
    output_path = Path(f"artifacts/traces/{baseline_name}_trace.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved trace to {output_path}")
    
    return df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Trace agent behavior')
    parser.add_argument('--agent', type=str, choices=['ppo', 'baseline'], required=True,
                       help='Agent type to trace')
    parser.add_argument('--ckpt', type=str, help='Checkpoint path for PPO agent')
    parser.add_argument('--baseline', type=str, choices=['AS', 'FIXED', 'RANDOM', 'INV'],
                       help='Baseline agent type')
    parser.add_argument('--config', type=str, default='configs/realistic_environment.yaml',
                       help='Configuration file')
    parser.add_argument('--steps', type=int, default=5000, help='Number of steps to trace')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    
    args = parser.parse_args()
    
    if args.agent == 'ppo':
        if not args.ckpt:
            print("Error: --ckpt required for PPO agent")
            return
        trace_ppo_agent(args.ckpt, args.config, args.steps, args.seed)
    elif args.agent == 'baseline':
        if not args.baseline:
            print("Error: --baseline required for baseline agent")
            return
        trace_baseline_agent(args.baseline, args.config, args.steps, args.seed)
    
    print("Trace evaluation completed!")


if __name__ == '__main__':
    main()
