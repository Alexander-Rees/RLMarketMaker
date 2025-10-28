"""
Enhanced Reward Calculator with Realistic Inventory Costs

Implements both quadratic and linear inventory penalties:
- Quadratic penalty: λ * inventory**2 (λ = 1e-3)
- Linear penalty: κ * abs(inventory) (κ = 1e-4)

Also includes volatility-aware risk adjustments.
"""

import numpy as np
from typing import Dict, Any


class EnhancedRewardCalculator:
    """Enhanced reward calculator with realistic inventory costs."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        # Inventory cost parameters
        self.lambda_inventory = config.get('lambda_inventory', 0.001)  # Quadratic penalty
        self.kappa_inventory = config.get('kappa_inventory', 0.0001)   # Linear penalty
        
        # Transaction costs
        self.transaction_fee = config.get('transaction_fee', 0.0002)
        
        # Risk controls
        self.max_inventory = config.get('max_inventory', 1000.0)
        self.max_loss_per_episode = config.get('max_loss_per_episode', 1000.0)
        self.kill_switch_threshold = config.get('kill_switch_threshold', -5000.0)
        
        # Volatility-aware risk scaling
        self.volatility_risk_scale = config.get('volatility_risk_scale', 1.0)
        
        # Soft position limits
        self.position_limit_threshold = config.get('position_limit_threshold', 25)
        self.position_limit_coeff = config.get('position_limit_coeff', 0.5)
        
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
                        midprice: float = 0.0,
                        volatility: float = 0.0) -> float:
        """
        Calculate enhanced reward with realistic inventory costs.
        
        Args:
            delta_pnl: Change in PnL from this step
            inventory: Current inventory position
            trade_size: Size of trade (if any)
            midprice: Current midprice (for fee calculation)
            volatility: Current market volatility
            
        Returns:
            Enhanced reward value
        """
        # Update episode tracking
        self.episode_pnl += delta_pnl
        if trade_size > 0:
            fee = trade_size * midprice * self.transaction_fee
            self.episode_fees += fee
            self.episode_trades += 1
        
        # Base reward components
        reward = delta_pnl
        
        # Enhanced inventory penalties
        # Quadratic penalty (inventory**2)
        inventory_quad_penalty = self.lambda_inventory * inventory**2
        
        # Linear penalty (abs(inventory))
        inventory_linear_penalty = self.kappa_inventory * abs(inventory)
        
        # Volatility-aware scaling
        volatility_scale = 1.0 + self.volatility_risk_scale * volatility
        
        # Apply inventory penalties with volatility scaling
        reward -= inventory_quad_penalty * volatility_scale
        reward -= inventory_linear_penalty * volatility_scale
        
        # Soft position limit penalty
        position_limit_threshold = getattr(self, 'position_limit_threshold', 25)
        position_limit_coeff = getattr(self, 'position_limit_coeff', 0.5)
        over_limit = max(0, abs(inventory) - position_limit_threshold)
        reward -= position_limit_coeff * (over_limit ** 2)
        
        # Transaction fees
        if trade_size > 0:
            fee = trade_size * midprice * self.transaction_fee
            reward -= fee
        
        # Risk controls with enhanced penalties
        if abs(inventory) > self.max_inventory:
            reward -= 200.0 * volatility_scale  # Large penalty for exceeding inventory limit
        
        if self.episode_pnl < self.kill_switch_threshold:
            reward -= 2000.0 * volatility_scale  # Kill switch penalty
        
        if self.episode_pnl < -self.max_loss_per_episode:
            reward -= 1000.0 * volatility_scale  # Per-episode loss limit penalty
        
        # Additional volatility-based risk penalty
        if volatility > 0.5:  # High volatility threshold
            reward -= 10.0 * volatility  # Additional penalty for high volatility
        
        return reward
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get enhanced episode statistics."""
        return {
            'episode_pnl': self.episode_pnl,
            'episode_fees': self.episode_fees,
            'episode_trades': self.episode_trades,
            'net_pnl': self.episode_pnl - self.episode_fees,
            'inventory_quad_penalty': self.lambda_inventory,
            'inventory_linear_penalty': self.kappa_inventory
        }
    
    def is_episode_terminated(self) -> bool:
        """Check if episode should be terminated due to risk controls."""
        return (abs(self.episode_pnl) > abs(self.kill_switch_threshold) or
                self.episode_pnl < -self.max_loss_per_episode)
