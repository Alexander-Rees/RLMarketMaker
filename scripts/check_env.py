#!/usr/bin/env python3
"""Environment API validation script for RLMarketMaker."""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.env_checker import check_env
from rlmarketmaker.env.realistic_market_env import RealisticMarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config


def validate_environment():
    """Validate environment API compatibility with SB3."""
    print("🔍 Validating RealisticMarketMakerEnv API...")
    
    # Load config
    config_path = "configs/realistic_environment.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    config = load_config(config_path)
    
    # Create environment
    feed = SyntheticFeed(seed=42)
    env = RealisticMarketMakerEnv(feed, config, seed=42)
    
    print(f"✅ Environment created successfully")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Run SB3 environment checker
    print("\n🔍 Running SB3 environment checker...")
    try:
        check_env(env, warn=True)
        print("✅ SB3 environment checker passed")
    except Exception as e:
        print(f"❌ SB3 environment checker failed: {e}")
        return False
    
    # Manual rollout test
    print("\n🔍 Running manual rollout test...")
    try:
        obs, info = env.reset()
        print(f"   Reset: obs shape={obs.shape}, obs dtype={obs.dtype}")
        
        for step in range(200):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Validate return types
            assert isinstance(obs, np.ndarray), f"obs should be np.ndarray, got {type(obs)}"
            assert obs.dtype == np.float32, f"obs should be float32, got {obs.dtype}"
            assert obs.shape == env.observation_space.shape, f"obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
            
            assert isinstance(reward, (int, float, np.number)), f"reward should be scalar, got {type(reward)}"
            assert isinstance(terminated, bool), f"terminated should be bool, got {type(terminated)}"
            assert isinstance(truncated, bool), f"truncated should be bool, got {type(truncated)}"
            assert isinstance(info, dict), f"info should be dict, got {type(info)}"
            
            # Check info values are JSON serializable
            for key, value in info.items():
                if isinstance(value, np.integer):
                    info[key] = int(value)
                elif isinstance(value, np.floating):
                    info[key] = float(value)
                elif isinstance(value, np.ndarray):
                    info[key] = value.tolist()
            
            if step % 50 == 0:
                print(f"   Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {step}")
                break
        
        print("✅ Manual rollout test passed")
        return True
        
    except Exception as e:
        print(f"❌ Manual rollout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    print("🚀 RLMarketMaker Environment API Validation")
    print("=" * 50)
    
    success = validate_environment()
    
    if success:
        print("\n✅ All environment API checks passed!")
        print("Environment is ready for SB3 training.")
    else:
        print("\n❌ Environment API validation failed!")
        print("Please fix the issues before proceeding with training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
