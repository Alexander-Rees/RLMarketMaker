#!/usr/bin/env python3
"""Environment performance benchmark script."""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config
from rlmarketmaker.utils.io import write_json


def benchmark_environment(config_path: str, steps: int, seed: int = 42) -> Dict[str, Any]:
    """
    Benchmark environment performance with no-op policy.
    
    Args:
        config_path: Path to config YAML
        steps: Number of steps to run
        seed: Random seed
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking environment for {steps} steps...")
    
    # Load config
    config = load_config(config_path)
    
    # Create environment
    feed = SyntheticFeed(seed=seed)
    env = RealisticMarketMakerEnv(feed, config, seed=seed)
    
    # No-op policy: always take middle action (no trading)
    noop_action = np.array([5, 5, 2])  # Middle offsets, medium size
    
    # Benchmark
    obs, _ = env.reset()
    start_time = time.time()
    
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(noop_action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate metrics
    steps_per_sec = steps / elapsed
    avg_ms_per_step = (elapsed / steps) * 1000
    
    results = {
        'steps': steps,
        'elapsed_seconds': elapsed,
        'steps_per_sec': steps_per_sec,
        'avg_ms_per_step': avg_ms_per_step,
        'seed': seed,
        'config': config_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Steps: {steps}")
    print(f"   Elapsed: {elapsed:.2f}s")
    print(f"   Steps/sec: {steps_per_sec:.2f}")
    print(f"   Avg ms/step: {avg_ms_per_step:.4f}")
    
    return results


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Benchmark environment performance')
    parser.add_argument('--config', type=str, default='configs/ppo_optimized.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--steps', type=int, default=200000,
                       help='Number of steps to benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='artifacts/benchmarks/env_benchmark.json',
                       help='Output JSON path')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_environment(args.config, args.steps, args.seed)
    
    # Write results
    write_json(args.output, results)
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == '__main__':
    main()

