#!/usr/bin/env python3
"""Evaluate agents on realistic market environment."""

import argparse
import os
import random
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Any, List

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.agents.baselines import (
    FixedSpreadStrategy, 
    RandomStrategy,
    InventoryMeanReversionStrategy,
    AvellanedaStoikovStrategy
)
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.utils.metrics import MarketMakingMetrics


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Using seed: {seed}")


def create_environment(config: Dict[str, Any], seed: int) -> RealisticMarketMakerEnv:
    """Create realistic market making environment."""
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    return env


def evaluate_agent(agent, env: RealisticMarketMakerEnv, episodes: int = 10) -> Dict[str, float]:
    """Evaluate a single agent."""
    metrics = MarketMakingMetrics()
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        inventory_history = []
        total_fills = 0
        total_orders = 0
        
        while not done and step < 2000:  # Max steps per episode
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
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track inventory
            inventory_history.append(info.get('inventory', 0))
            
            # Track fills (simplified - assume any non-zero reward means a fill)
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


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate agents on realistic environment')
    parser.add_argument('--config', type=str, default='configs/realistic_environment.yaml',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create environment
    env = create_environment(config, args.seed)
    
    # Define agents
    agents = {
        'FixedSpread': FixedSpreadStrategy(spread_ticks=2, max_inventory=300.0),
        'Random': RandomStrategy(seed=args.seed),
        'InventoryMeanReversion': InventoryMeanReversionStrategy(base_spread=2, max_inventory=300.0),
        'AvellanedaStoikov': AvellanedaStoikovStrategy(max_inventory=300.0)
    }
    
    print(f"Evaluating agents with {args.episodes} episodes each...")
    
    # Evaluate each agent
    results = {}
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name} baseline")
        agent_results = evaluate_agent(agent, env, args.episodes)
        results[agent_name] = agent_results
        
        print(f"{agent_name} Results: PnL={agent_results['total_pnl']:.2f}, "
              f"Sharpe={agent_results['sharpe_ratio']:.2f}")
    
    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results to JSON-serializable format
    json_results = {}
    for agent_name, agent_results in results.items():
        json_results[agent_name] = {k: convert_numpy(v) for k, v in agent_results.items()}
    
    os.makedirs('logs', exist_ok=True)
    with open('logs/realistic_evaluation_report.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("Comparison report saved to logs/realistic_evaluation_report.json")
    
    # Print summary
    print("\n" + "="*50)
    print("REALISTIC ENVIRONMENT EVALUATION SUMMARY")
    print("="*50)
    
    # Find best performers
    best_pnl = max(results.items(), key=lambda x: x[1]['total_pnl'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    best_fill_rate = max(results.items(), key=lambda x: x[1]['mean_fill_rate'])
    
    print(f"Best PnL: {best_pnl[0]} ({best_pnl[1]['total_pnl']:.2f})")
    print(f"Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")
    print(f"Best Fill Rate: {best_fill_rate[0]} ({best_fill_rate[1]['mean_fill_rate']:.2f})")
    
    # Check if random strategy is still dominant
    random_pnl = results['Random']['total_pnl']
    smart_strategies = [results[name]['total_pnl'] for name in ['FixedSpread', 'InventoryMeanReversion', 'AvellanedaStoikov']]
    best_smart = max(smart_strategies)
    
    print(f"\nRandom Strategy PnL: {random_pnl:.2f}")
    print(f"Best Smart Strategy PnL: {best_smart:.2f}")
    
    if random_pnl > best_smart:
        print("⚠️  Random strategy still dominant - environment needs further tuning")
    else:
        print("✅ Realistic environment successfully reduces random strategy dominance!")
    
    print(f"\nDetailed results saved to: logs/realistic_evaluation_report.json")
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
