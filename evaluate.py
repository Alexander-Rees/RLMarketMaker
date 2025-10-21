#!/usr/bin/env python3
"""Evaluation script for RL Market Maker."""

import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from rlmarketmaker.env.market_env import MarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.agents.baselines import (
    FixedSpreadStrategy, 
    AvellanedaStoikovStrategy, 
    RandomStrategy,
    InventoryMeanReversionStrategy
)
from rlmarketmaker.utils.config import load_config, get_config_path
from rlmarketmaker.utils.metrics import MarketMakingMetrics
from rlmarketmaker.utils.logging import EvaluationLogger


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_random_seed(seed)


def create_env(config: dict, seed: int = None):
    """Create market making environment."""
    feed = SyntheticFeed(seed=seed)
    env = MarketMakerEnv(feed, config, seed=seed)
    return env


def load_ppo_model(model_path: str, env):
    """Load trained PPO model."""
    model = PPO.load(model_path, env=env)
    return model


def evaluate_agent(agent, env, config: dict, num_episodes: int = 10, seed: int = 42) -> Dict[str, float]:
    """Evaluate an agent on the environment."""
    set_seeds(seed)
    
    metrics = MarketMakingMetrics()
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        inventory_history = []
        
        done = False
        while not done:
            # Get action from agent
            if hasattr(agent, 'predict'):
                # RL agent
                action, _ = agent.predict(obs, deterministic=True)
            else:
                # Baseline strategy
                market_state = {
                    'midprice': env.current_tick.midprice,
                    'spread': env.current_tick.spread
                }
                action = agent.get_action(obs, market_state)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            inventory_history.append(info['inventory'])
            
            done = terminated or truncated
        
        # Calculate episode metrics
        episode_pnl = info['total_pnl']
        fill_rate = 0.8  # Placeholder - would need to track actual fills
        max_dd = max(inventory_history) - min(inventory_history) if inventory_history else 0
        
        # Add to metrics
        metrics.add_episode(
            episode_pnl=episode_pnl,
            inventory_history=inventory_history,
            fill_rate=fill_rate,
            episode_length=episode_length
        )
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Calculate summary metrics
    summary_metrics = metrics.calculate_summary_metrics()
    
    # Add additional metrics
    summary_metrics.update({
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'total_episodes': num_episodes
    })
    
    return summary_metrics


def evaluate_all_agents(config: dict, model_path: str = None, num_episodes: int = 10, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Evaluate all agents and return comparison results."""
    results = {}
    
    # Create environment
    env = create_env(config, seed=seed)
    
    # Evaluate PPO agent if model path provided
    if model_path and os.path.exists(model_path):
        print(f"Evaluating PPO agent from {model_path}")
        ppo_model = load_ppo_model(model_path, env)
        ppo_metrics = evaluate_agent(ppo_model, env, config, num_episodes, seed)
        results['PPO'] = ppo_metrics
        print(f"PPO Results: PnL={ppo_metrics['total_pnl']:.2f}, Sharpe={ppo_metrics['sharpe_ratio']:.2f}")
    
    # Evaluate baseline strategies
    baseline_configs = {
        'FixedSpread': {'strategy_name': 'fixed_spread', 'spread_ticks': 2, 'max_inventory': 100.0},
        'AvellanedaStoikov': {'strategy_name': 'avellaneda_stoikov', 'gamma': 0.1, 'sigma': 0.2, 'max_inventory': 100.0},
        'Random': {'strategy_name': 'random', 'seed': seed},
        'InventoryMeanReversion': {'strategy_name': 'inventory_mean_reversion', 'base_spread': 2, 'max_inventory': 100.0}
    }
    
    for name, baseline_config in baseline_configs.items():
        print(f"Evaluating {name} baseline")
        
        # Create baseline strategy
        from rlmarketmaker.agents.baselines import create_baseline_strategy
        baseline_strategy = create_baseline_strategy(
            baseline_config['strategy_name'], 
            baseline_config
        )
        
        # Evaluate
        baseline_metrics = evaluate_agent(baseline_strategy, env, config, num_episodes, seed)
        results[name] = baseline_metrics
        
        print(f"{name} Results: PnL={baseline_metrics['total_pnl']:.2f}, Sharpe={baseline_metrics['sharpe_ratio']:.2f}")
    
    return results


def create_comparison_report(results: Dict[str, Dict[str, float]], output_path: str):
    """Create comparison report."""
    report = {
        'timestamp': str(Path(output_path).stem),
        'summary': {},
        'detailed_results': results
    }
    
    # Create summary comparison
    if results:
        # Find best performing agent for each metric
        best_pnl = max(results.items(), key=lambda x: x[1].get('total_pnl', -float('inf')))
        best_sharpe = max(results.items(), key=lambda x: x[1].get('sharpe_ratio', -float('inf')))
        best_fill_rate = max(results.items(), key=lambda x: x[1].get('mean_fill_rate', -float('inf')))
        
        report['summary'] = {
            'best_pnl': {'agent': best_pnl[0], 'value': best_pnl[1].get('total_pnl', 0)},
            'best_sharpe': {'agent': best_sharpe[0], 'value': best_sharpe[1].get('sharpe_ratio', 0)},
            'best_fill_rate': {'agent': best_fill_rate[0], 'value': best_fill_rate[1].get('mean_fill_rate', 0)},
            'total_agents': len(results)
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comparison report saved to {output_path}")
    return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate RL Market Maker')
    parser.add_argument('--config', type=str, default='configs/ppo.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained PPO model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='logs/evaluation_report.json',
                       help='Output path for evaluation report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"Evaluating agents with {args.episodes} episodes each...")
    print(f"Using seed: {args.seed}")
    
    # Evaluate all agents
    results = evaluate_all_agents(
        config=config,
        model_path=args.model,
        num_episodes=args.episodes,
        seed=args.seed
    )
    
    # Create comparison report
    report = create_comparison_report(results, str(output_path))
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if report['summary']:
        print(f"Best PnL: {report['summary']['best_pnl']['agent']} ({report['summary']['best_pnl']['value']:.2f})")
        print(f"Best Sharpe: {report['summary']['best_sharpe']['agent']} ({report['summary']['best_sharpe']['value']:.2f})")
        print(f"Best Fill Rate: {report['summary']['best_fill_rate']['agent']} ({report['summary']['best_fill_rate']['value']:.2f})")
    
    print(f"\nDetailed results saved to: {output_path}")
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
