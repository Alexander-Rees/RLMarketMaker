#!/usr/bin/env python3
"""Backtest CLI for agent evaluation with trace output."""

import argparse
import sys
import time
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.agents.min_ppo import MinPPO
from rlmarketmaker.agents.baselines import (
    FixedSpreadStrategy, RandomStrategy, InventoryMeanReversionStrategy, AvellanedaStoikovStrategy
)
from rlmarketmaker.utils.io import write_json, ensure_dir


def create_agent(agent_type: str, checkpoint_path: str, config: Dict[str, Any], env):
    """Create agent instance based on type."""
    if agent_type == 'ppo':
        if not checkpoint_path:
            raise ValueError("--ckpt required for PPO agent")
        
        state_dim = env.observation_space.shape[0]
        action_dims = env.action_space.nvec.tolist()
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
        
        # Load checkpoint (remove .pt extension if present)
        ckpt = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
        trainer.load(ckpt)
        return trainer, 'ppo'
    
    elif agent_type == 'as':
        return AvellanedaStoikovStrategy(), 'baseline'
    elif agent_type == 'fixed':
        return FixedSpreadStrategy(), 'baseline'
    elif agent_type == 'inv':
        return InventoryMeanReversionStrategy(), 'baseline'
    elif agent_type == 'random':
        return RandomStrategy(), 'baseline'
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_backtest(
    agent_type: str,
    checkpoint_path: str,
    config_path: str,
    steps: int,
    seed: int
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run backtest and return metrics and trace.
    
    Returns:
        Tuple of (metrics_dict, trace_dataframe)
    """
    # Load config
    config = load_config(config_path)
    
    # Create environment
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    
    # Create agent
    agent, agent_category = create_agent(agent_type, checkpoint_path, config, env)
    
    # Run backtest
    obs, _ = env.reset()
    trace_data = []
    episode_pnls = []
    episode_rewards = []
    inventory_history = []
    total_fills = 0
    total_orders = 0
    
    for step in range(steps):
        # Get action
        if agent_category == 'ppo':
            norm_obs = agent.vec_normalize.normalize(obs.reshape(1, -1)).squeeze()
            action, _ = agent.get_action(norm_obs, deterministic=True)
        else:
            market_state = {
                'midprice': env.current_tick.midprice,
                'spread': env.current_tick.spread,
                'bid': env.current_tick.midprice - env.current_tick.spread/2,
                'ask': env.current_tick.midprice + env.current_tick.spread/2
            }
            action = agent.get_action(obs, market_state)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Record trace
        trace_row = {
            'step': step,
            'midprice': info.get('midprice', 0),
            'bid_quote': info.get('bid_quote', 0),
            'ask_quote': info.get('ask_quote', 0),
            'filled_bid': int(info.get('filled_bid', 0)),
            'filled_ask': int(info.get('filled_ask', 0)),
            'inventory': info.get('inventory', 0),
            'pnl': info.get('total_pnl', 0),
            'cumulative_pnl': info.get('cumulative_pnl', 0),
            'spread': info.get('ask_quote', 0) - info.get('bid_quote', 0),
            'volatility': info.get('volatility', 0)
        }
        trace_data.append(trace_row)
        
        # Track metrics
        inventory_history.append(info.get('inventory', 0))
        total_orders += 1
        if info.get('filled_bid', 0) or info.get('filled_ask', 0):
            total_fills += 1
        
        obs = next_obs
        
        if terminated or truncated:
            episode_pnls.append(info.get('total_pnl', 0))
            episode_rewards.append(reward)
            obs, _ = env.reset()
    
    # Compute metrics
    if episode_pnls:
        mean_pnl = np.mean(episode_pnls)
        std_pnl = np.std(episode_pnls)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    else:
        mean_pnl = info.get('total_pnl', 0)
        sharpe = 0.0
    
    fill_rate = total_fills / total_orders if total_orders > 0 else 0.0
    inv_var = np.var(inventory_history) if inventory_history else 0.0
    
    # Calculate max drawdown
    if trace_data and len(trace_data) > 1:
        pnls = [row['pnl'] for row in trace_data]
        cumulative = np.cumsum([pnl - (pnls[i-1] if i > 0 else 0) for i, pnl in enumerate(pnls)])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    else:
        max_dd = 0.0
    
    metrics = {
        'agent_type': agent_type,
        'steps': steps,
        'seed': seed,
        'pnl': float(mean_pnl),
        'pnl_std': float(std_pnl) if episode_pnls else 0.0,
        'sharpe': float(sharpe),
        'fill_rate': float(fill_rate),
        'inv_var': float(inv_var),
        'max_drawdown': float(max_dd),
        'total_fills': total_fills,
        'total_orders': total_orders
    }
    
    trace_df = pd.DataFrame(trace_data)
    
    return metrics, trace_df


def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(description='Run agent backtest with trace output')
    parser.add_argument('--agent', type=str, 
                       choices=['ppo', 'as', 'fixed', 'inv', 'random'],
                       required=True, help='Agent type')
    parser.add_argument('--ckpt', type=str, help='Checkpoint path (required for ppo)')
    parser.add_argument('--config', type=str, default='configs/ppo_optimized.yaml',
                       help='Config YAML file')
    parser.add_argument('--steps', type=int, default=50000, help='Number of steps')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='artifacts/backtests',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.agent == 'ppo' and not args.ckpt:
        print("Error: --ckpt required for PPO agent")
        return
    
    # Generate run ID
    run_id = f"{args.agent}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running backtest: {args.agent} agent")
    print(f"Config: {args.config}")
    print(f"Steps: {args.steps}, Seed: {args.seed}")
    print(f"Output: {output_dir}")
    
    # Run backtest
    metrics, trace_df = run_backtest(
        args.agent, args.ckpt, args.config, args.steps, args.seed
    )
    
    # Save outputs
    write_json(str(output_dir / 'metrics.json'), metrics)
    trace_df.to_csv(str(output_dir / 'trace.csv'), index=False)
    
    # Print summary
    print(f"\n📊 Backtest Results:")
    print(f"   PnL: {metrics['pnl']:.2f} ± {metrics['pnl_std']:.2f}")
    print(f"   Sharpe: {metrics['sharpe']:.2f}")
    print(f"   Fill Rate: {metrics['fill_rate']:.2%}")
    print(f"   Inv Var: {metrics['inv_var']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}")
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

