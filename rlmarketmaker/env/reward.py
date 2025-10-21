"""Reward calculation for market making environment."""

import numpy as np
from typing import Dict, Any


class RewardCalculator:
    """Calculate rewards for market making actions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        self.lambda_inventory = config.get('lambda_inventory', 0.01)
        self.transaction_fee = config.get('transaction_fee', 0.0001)
        self.max_inventory = config.get('max_inventory', 1000.0)
        self.max_loss_per_episode = config.get('max_loss_per_episode', 1000.0)
        self.kill_switch_threshold = config.get('kill_switch_threshold', -5000.0)
        
        # Track episode state
        self.episode_pnl = 0.0
        self.episode_fees = 0.0
        self.episode_trades = 0
        
    def reset(self):
        """Reset episode tracking."""
        self.episode_pnl = 0.0
        self.episode_fees = 0.0
        self.episode_trades = 0
    
    def calculate_reward(self, 
                        delta_pnl: float,
                        inventory: float,
                        trade_size: float = 0.0,
                        midprice: float = 0.0) -> float:
        """
        Calculate reward for a single step.
        
        Args:
            delta_pnl: Change in PnL from this step
            inventory: Current inventory position
            trade_size: Size of trade (if any)
            midprice: Current midprice (for fee calculation)
            
        Returns:
            Reward value
        """
        # Update episode tracking
        self.episode_pnl += delta_pnl
        if trade_size > 0:
            fee = trade_size * midprice * self.transaction_fee
            self.episode_fees += fee
            self.episode_trades += 1
        
        # Base reward components
        reward = delta_pnl
        
        # Inventory penalty (quadratic)
        inventory_penalty = self.lambda_inventory * inventory**2
        reward -= inventory_penalty
        
        # Transaction fees
        if trade_size > 0:
            fee = trade_size * midprice * self.transaction_fee
            reward -= fee
        
        # Risk controls
        if abs(inventory) > self.max_inventory:
            reward -= 100.0  # Large penalty for exceeding inventory limit
        
        if self.episode_pnl < self.kill_switch_threshold:
            reward -= 1000.0  # Kill switch penalty
        
        if self.episode_pnl < -self.max_loss_per_episode:
            reward -= 500.0  # Per-episode loss limit penalty
        
        return reward
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        return {
            'episode_pnl': self.episode_pnl,
            'episode_fees': self.episode_fees,
            'episode_trades': self.episode_trades,
            'net_pnl': self.episode_pnl - self.episode_fees
        }
    
    def is_episode_terminated(self) -> bool:
        """Check if episode should be terminated due to risk controls."""
        return (abs(self.episode_pnl) > self.max_loss_per_episode or 
                self.episode_pnl < self.kill_switch_threshold)


class InventoryPenalty:
    """Inventory-based penalty calculation."""
    
    def __init__(self, lambda_inv: float = 0.01, max_inv: float = 1000.0):
        self.lambda_inv = lambda_inv
        self.max_inv = max_inv
    
    def penalty(self, inventory: float) -> float:
        """Calculate inventory penalty."""
        if abs(inventory) > self.max_inv:
            return 100.0  # Hard limit penalty
        return self.lambda_inv * inventory**2
    
    def is_over_limit(self, inventory: float) -> bool:
        """Check if inventory exceeds limit."""
        return abs(inventory) > self.max_inv


class FeeCalculator:
    """Transaction fee calculation."""
    
    def __init__(self, fee_rate: float = 0.0001):
        self.fee_rate = fee_rate
    
    def calculate_fee(self, trade_size: float, price: float) -> float:
        """Calculate transaction fee."""
        return trade_size * price * self.fee_rate
    
    def calculate_fees(self, trades: list) -> float:
        """Calculate total fees for a list of trades."""
        total_fees = 0.0
        for trade in trades:
            total_fees += self.calculate_fee(trade['size'], trade['price'])
        return total_fees
