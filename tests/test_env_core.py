"""Core environment tests."""

import pytest
import numpy as np
from rlmarketmaker.env.market_env import MarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config


class TestMarketMakerEnv:
    """Test market making environment core functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Load synthetic config
        config = load_config("configs/synthetic.yaml")
        config.update({
            'episode_length': 100,
            'max_inventory': 100.0,
            'max_loss_per_episode': 1000.0,
            'kill_switch_threshold': -5000.0,
            'lambda_inventory': 0.01,
            'transaction_fee': 0.0001
        })
        
        # Create environment
        feed = SyntheticFeed(seed=42)
        self.env = MarketMakerEnv(feed, config, seed=42)
    
    def test_pnl_invariant(self):
        """Test that cash + inventory * midprice ≈ total PnL."""
        obs, _ = self.env.reset()
        
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Check PnL invariant
            cash = info['cash']
            inventory = info['inventory']
            total_pnl = info['total_pnl']
            midprice = self.env.current_tick.midprice
            
            expected_pnl = cash + inventory * midprice
            assert abs(total_pnl - expected_pnl) < 1e-6, f"PnL invariant violated: {total_pnl} != {expected_pnl}"
            
            if terminated or truncated:
                break
    
    def test_reward_calculation(self):
        """Test reward calculation components."""
        obs, _ = self.env.reset()
        
        # Take a simple action
        action = np.array([0, 0, 0])  # At touch, small size
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Reward should include PnL change, inventory penalty, and fees
        assert isinstance(reward, float), "Reward should be a float"
        
        # Check that reward is reasonable (not NaN or infinite)
        assert np.isfinite(reward), f"Reward should be finite, got {reward}"
    
    def test_latency_delay(self):
        """Test that latency actually delays actions."""
        # Set latency to 3 steps
        self.env.latency = 3
        
        obs, _ = self.env.reset()
        
        # Take an action
        action = np.array([1, 1, 1])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Action should be in latency queue, not executed yet
        assert len(self.env.latency_queue) == 1, "Action should be in latency queue"
        
        # Step forward to execute the action (latency = 3, so need 3 more steps)
        for _ in range(3):
            obs, reward, terminated, truncated, info = self.env.step(np.array([0, 0, 0]))
            if terminated or truncated:
                break
        
        # Action should be executed (queue should have 4 actions: 3 new ones + 1 more from the last step)
        assert len(self.env.latency_queue) == 4, "Should have 4 actions in queue after execution"
    
    def test_fill_probability_monotonic(self):
        """Test that fill probability decreases with distance."""
        fill_model = self.env.fill_model
        
        distances = np.linspace(0, 10, 11)
        probs = [fill_model.fill_probability(d) for d in distances]
        
        # Check monotonicity
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i-1], f"Fill probability should be monotonic: {probs[i]} > {probs[i-1]}"
        
        # Check that probability at distance 0 is highest
        assert probs[0] == max(probs), "Fill probability should be highest at distance 0"
    
    def test_reset_no_state_leak(self):
        """Test that environment reset doesn't leak state between episodes."""
        # First episode
        obs1, _ = self.env.reset()
        initial_inventory = self.env.inventory
        initial_cash = self.env.cash
        
        # Take some actions
        for _ in range(5):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        
        # Reset environment
        obs2, _ = self.env.reset()
        
        # Check that state is reset
        assert self.env.inventory == initial_inventory, "Inventory should be reset"
        assert self.env.cash == initial_cash, "Cash should be reset"
        assert self.env.step_count == 0, "Step count should be reset"
        assert len(self.env.latency_queue) == 0, "Latency queue should be empty"
    
    def test_risk_controls(self):
        """Test risk control mechanisms."""
        obs, _ = self.env.reset()
        
        # Test inventory limit
        self.env.inventory = self.env.max_inventory + 1
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Should be terminated due to inventory limit
        assert terminated, "Environment should terminate when inventory exceeds limit"
    
    def test_observation_space(self):
        """Test observation space properties."""
        obs, _ = self.env.reset()
        
        # Check observation shape
        assert obs.shape == (5,), f"Observation should have shape (5,), got {obs.shape}"
        
        # Check observation is finite
        assert np.all(np.isfinite(obs)), f"Observation should be finite, got {obs}"
        
        # Check observation space bounds
        assert self.env.observation_space.contains(obs), "Observation should be within bounds"
    
    def test_action_space(self):
        """Test action space properties."""
        # Test valid actions
        for _ in range(10):
            action = self.env.action_space.sample()
            assert self.env.action_space.contains(action), f"Action should be valid: {action}"
        
        # Test action bounds
        assert np.all(action >= 0), "Action values should be non-negative"
        assert action[0] < 10, "Bid offset should be < 10"
        assert action[1] < 10, "Ask offset should be < 10"
        assert action[2] < 5, "Size index should be < 5"
