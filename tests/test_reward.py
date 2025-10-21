"""Reward calculation tests."""

import pytest
import numpy as np
from rlmarketmaker.env.reward import RewardCalculator, InventoryPenalty, FeeCalculator


class TestRewardCalculator:
    """Test reward calculation functionality."""
    
    def setup_method(self):
        """Set up test reward calculator."""
        config = {
            'lambda_inventory': 0.01,
            'transaction_fee': 0.0001,
            'max_inventory': 100.0,
            'max_loss_per_episode': 1000.0,
            'kill_switch_threshold': -5000.0
        }
        self.reward_calc = RewardCalculator(config)
    
    def test_reward_components(self):
        """Test individual reward components."""
        # Test PnL component
        reward = self.reward_calc.calculate_reward(
            delta_pnl=100.0,
            inventory=0.0,
            trade_size=0.0,
            midprice=100.0
        )
        assert reward == 100.0, f"Reward should equal delta PnL, got {reward}"
        
        # Test inventory penalty
        reward = self.reward_calc.calculate_reward(
            delta_pnl=0.0,
            inventory=10.0,
            trade_size=0.0,
            midprice=100.0
        )
        expected_penalty = 0.01 * 10**2  # lambda * inventory^2
        assert reward == -expected_penalty, f"Inventory penalty incorrect: {reward} != {-expected_penalty}"
        
        # Test transaction fees
        reward = self.reward_calc.calculate_reward(
            delta_pnl=0.0,
            inventory=0.0,
            trade_size=100.0,
            midprice=100.0
        )
        expected_fee = 100.0 * 100.0 * 0.0001  # size * price * fee_rate
        assert reward == -expected_fee, f"Transaction fee incorrect: {reward} != {-expected_fee}"
    
    def test_risk_controls(self):
        """Test risk control penalties."""
        # Test inventory limit
        reward = self.reward_calc.calculate_reward(
            delta_pnl=0.0,
            inventory=150.0,  # Exceeds max_inventory
            trade_size=0.0,
            midprice=100.0
        )
        assert reward < -50.0, f"Inventory limit penalty too small: {reward}"
        
        # Test kill switch
        self.reward_calc.episode_pnl = -6000.0  # Below kill switch threshold
        reward = self.reward_calc.calculate_reward(
            delta_pnl=0.0,
            inventory=0.0,
            trade_size=0.0,
            midprice=100.0
        )
        assert reward < -500.0, f"Kill switch penalty too small: {reward}"
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        # Test normal case
        assert not self.reward_calc.is_episode_terminated(), "Should not terminate normally"
        
        # Test loss limit
        self.reward_calc.episode_pnl = -1500.0  # Exceeds max loss
        assert self.reward_calc.is_episode_terminated(), "Should terminate due to loss limit"
        
        # Test kill switch
        self.reward_calc.episode_pnl = -6000.0  # Below kill switch
        assert self.reward_calc.is_episode_terminated(), "Should terminate due to kill switch"
    
    def test_episode_stats(self):
        """Test episode statistics tracking."""
        # Reset and take some actions
        self.reward_calc.reset()
        
        # Simulate some trading
        self.reward_calc.calculate_reward(100.0, 0.0, 50.0, 100.0)  # PnL + trade
        self.reward_calc.calculate_reward(50.0, 10.0, 25.0, 105.0)  # More PnL + trade
        
        stats = self.reward_calc.get_episode_stats()
        
        assert stats['episode_pnl'] == 150.0, f"Episode PnL incorrect: {stats['episode_pnl']}"
        assert stats['episode_trades'] == 2, f"Episode trades incorrect: {stats['episode_trades']}"
        assert stats['episode_fees'] > 0, "Episode fees should be positive"
        assert stats['net_pnl'] < stats['episode_pnl'], "Net PnL should be less than gross PnL"


class TestInventoryPenalty:
    """Test inventory penalty calculation."""
    
    def setup_method(self):
        """Set up test inventory penalty."""
        self.penalty = InventoryPenalty(lambda_inv=0.01, max_inv=100.0)
    
    def test_penalty_calculation(self):
        """Test penalty calculation."""
        # Test normal inventory
        penalty = self.penalty.penalty(10.0)
        assert penalty == 0.01 * 10**2, f"Penalty incorrect: {penalty}"
        
        # Test zero inventory
        penalty = self.penalty.penalty(0.0)
        assert penalty == 0.0, f"Zero inventory should have zero penalty: {penalty}"
        
        # Test negative inventory
        penalty = self.penalty.penalty(-10.0)
        assert penalty == 0.01 * 10**2, f"Negative inventory penalty incorrect: {penalty}"
    
    def test_inventory_limit(self):
        """Test inventory limit checking."""
        # Test within limit
        assert not self.penalty.is_over_limit(50.0), "Should be within limit"
        assert not self.penalty.is_over_limit(-50.0), "Should be within limit"
        
        # Test over limit
        assert self.penalty.is_over_limit(150.0), "Should be over limit"
        assert self.penalty.is_over_limit(-150.0), "Should be over limit"


class TestFeeCalculator:
    """Test fee calculation functionality."""
    
    def setup_method(self):
        """Set up test fee calculator."""
        self.fee_calc = FeeCalculator(fee_rate=0.0001)
    
    def test_fee_calculation(self):
        """Test fee calculation."""
        # Test single trade
        fee = self.fee_calc.calculate_fee(100.0, 50.0)
        expected = 100.0 * 50.0 * 0.0001
        assert fee == expected, f"Fee calculation incorrect: {fee} != {expected}"
        
        # Test zero size
        fee = self.fee_calc.calculate_fee(0.0, 50.0)
        assert fee == 0.0, f"Zero size should have zero fee: {fee}"
        
        # Test zero price
        fee = self.fee_calc.calculate_fee(100.0, 0.0)
        assert fee == 0.0, f"Zero price should have zero fee: {fee}"
    
    def test_multiple_trades(self):
        """Test fee calculation for multiple trades."""
        trades = [
            {'size': 100.0, 'price': 50.0},
            {'size': 200.0, 'price': 60.0},
            {'size': 50.0, 'price': 40.0}
        ]
        
        total_fee = self.fee_calc.calculate_fees(trades)
        
        expected = (100.0 * 50.0 + 200.0 * 60.0 + 50.0 * 40.0) * 0.0001
        assert total_fee == expected, f"Total fee incorrect: {total_fee} != {expected}"
