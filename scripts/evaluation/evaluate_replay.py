#!/usr/bin/env python3
"""
Evaluate PPO agent vs baselines on historical replay data.

This script loads a trained PPO model and compares its performance
against baseline strategies on historical market data.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlmarketmaker.env.replay_market_env import ReplayMarketMakerEnv
from rlmarketmaker.data.feeds import PolygonReplayFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO
from rlmarketmaker.agents.baselines import (
    FixedSpreadStrategy, RandomStrategy, InventoryMeanReversionStrategy, AvellanedaStoikovStrategy
)
from rlmarketmaker.utils.metrics import MarketMakingMetrics


def evaluate_agent_replay(agent, env: ReplayMarketMakerEnv, episodes: int = 10) -> Dict[str, float]:
    """Evaluate a single agent on replay data."""
    metrics = MarketMakingMetrics()
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        inventory_history = []
        total_fills = 0
        total_orders = 0
        episode_rewards = 0.0
        
        while not done and step < 2000:  # Max steps per episode
            # Get market state for agent
            market_state = {
                'midprice': env.current_tick.midprice,
                'spread': env.current_tick.spread,
                'bid': env.current_tick.midprice - env.current_tick.spread/2,
                'ask': env.current_tick.midprice + env.current_tick.spread/2
            }
            
            # Get action from agent
            if hasattr(agent, 'get_action') and hasattr(agent, 'vec_normalize'):
                # PPO agent
                result = agent.get_action(obs, deterministic=True)
                if len(result) == 2:
                    action, _ = result
                else:
                    action, _, _ = result
            else:
                # Baseline agent
                action = agent.get_action(obs, market_state)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track metrics
            episode_rewards += reward
            inventory_history.append(info.get('inventory', 0))
            total_orders += 1
            if reward != 0:
                total_fills += 1
            
            done = terminated or truncated
            step += 1
        
        # Calculate episode metrics
        episode_pnl = info.get('total_pnl', 0)
        fill_rate = total_fills / total_orders if total_orders > 0 else 0.0
        
        # Add episode to metrics
        metrics.add_episode(
            episode_pnl=episode_pnl,
            inventory_history=inventory_history,
            fill_rate=fill_rate,
            episode_length=step
        )
    
    return metrics.calculate_summary_metrics()


def evaluate_ppo_replay(checkpoint_path: str, config_path: str, episodes: int = 10) -> Dict[str, float]:
    """Evaluate PPO agent on replay data."""
    print(f"Evaluating PPO agent from {checkpoint_path}...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    feed = PolygonReplayFeed(seed=42)
    env = ReplayMarketMakerEnv(feed, config['env'], seed=42)
    
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
    
    # Evaluate
    results = evaluate_agent_replay(trainer, env, episodes)
    env.close()
    
    return results


def evaluate_baseline_replay(baseline_name: str, config_path: str, episodes: int = 10) -> Dict[str, float]:
    """Evaluate baseline agent on replay data."""
    print(f"Evaluating {baseline_name} baseline agent...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    feed = PolygonReplayFeed(seed=42)
    env = ReplayMarketMakerEnv(feed, config['env'], seed=42)
    
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
    
    # Evaluate
    results = evaluate_agent_replay(agent, env, episodes)
    env.close()
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate agents on replay data')
    parser.add_argument('--config', type=str, default='configs/polygon_replay.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='logs/checkpoints/policy.pt',
                       help='Path to PPO checkpoint')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--output', type=str, default='logs/replay_evaluation.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("🚀 RLMarketMaker Replay Evaluation")
    print("==================================")
    
    # Evaluate PPO agent
    print("\n📊 Evaluating PPO Agent...")
    ppo_results = evaluate_ppo_replay(args.checkpoint, args.config, args.episodes)
    
    # Evaluate baseline agents
    baselines = ['AS', 'FIXED', 'RANDOM', 'INV']
    baseline_results = {}
    
    for baseline in baselines:
        print(f"\n📊 Evaluating {baseline} Baseline...")
        try:
            baseline_results[baseline] = evaluate_baseline_replay(baseline, args.config, args.episodes)
        except Exception as e:
            print(f"Error evaluating {baseline}: {e}")
            baseline_results[baseline] = {}
    
    # Compile results
    results = {
        'PPO_RL_Replay': ppo_results,
        **{f'{baseline}_Replay': baseline_results[baseline] for baseline in baselines}
    }
    
    # Print summary
    print("\n📈 Replay Evaluation Results:")
    print("=" * 50)
    
    for agent_name, agent_results in results.items():
        if agent_results:
            print(f"\n{agent_name}:")
            print(f"  Mean PnL: {agent_results.get('mean_episode_pnl', 0):.2f}")
            print(f"  Sharpe Ratio: {agent_results.get('sharpe_ratio', 0):.2f}")
            print(f"  Mean Fill Rate: {agent_results.get('mean_fill_rate', 0):.3f}")
            print(f"  Inventory Variance: {agent_results.get('inventory_variance', 0):.2f}")
            print(f"  Max Drawdown: {agent_results.get('max_drawdown', 0):.2f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")
    
    # Find best performing agent
    best_agent = max(results.items(), key=lambda x: x[1].get('mean_episode_pnl', -float('inf')))
    print(f"\n🏆 Best Agent: {best_agent[0]} (PnL: {best_agent[1].get('mean_episode_pnl', 0):.2f})")


if __name__ == '__main__':
    main()
