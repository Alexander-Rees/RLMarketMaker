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
        Initialize enhanced reward calculator with adaptive risk penalties.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        # Base inventory penalty
        self.lambda_base = config.get('lambda_inventory', 0.0025)
        
        # Adaptive penalty parameters
        self.volatility_alpha = config.get('volatility_alpha', 0.3)
        self.kappa_multiplier = config.get('kappa_multiplier', 3.0)
        self.tau_multiplier = config.get('tau_multiplier', 0.5)
        
        # Soft threshold
        self.H = config.get('position_limit_threshold', 30)
        
        # Transaction costs
        self.transaction_fee = config.get('transaction_fee', 0.0002)
        
        # Risk controls
        self.max_inventory = config.get('max_inventory', 1000.0)
        self.max_loss_per_episode = config.get('max_loss_per_episode', 1000.0)
        self.kill_switch_threshold = config.get('kill_switch_threshold', -5000.0)
        
        # Track episode state
        self.episode_pnl = 0.0
        self.episode_fees = 0.0
        self.episode_trades = 0
        self.episode_spread_captured = 0.0
        
        # Volatility tracking
        self.volatility_history = []
        self.volatility_mean = 0.0
        self.volatility_std = 1.0
        
    def reset(self):
        """Reset episode tracking."""
        self.episode_pnl = 0.0
        self.episode_fees = 0.0
        self.episode_trades = 0
        self.episode_spread_captured = 0.0
    
    def calculate_reward(self, 
                        delta_pnl: float,
                        inventory: float,
                        trade_size: float = 0.0,
                        midprice: float = 0.0,
                        volatility: float = 0.0,
                        spread_captured: float = 0.0) -> float:
        """
        Calculate adaptive reward with volatility-aware inventory penalties.
        
        Args:
            delta_pnl: Change in PnL from this step
            inventory: Current inventory position
            trade_size: Size of trade (if any)
            midprice: Current midprice (for fee calculation)
            volatility: Current market volatility
            spread_captured: Spread captured from fills
            
        Returns:
            Enhanced reward value
        """
        # Update episode tracking
        self.episode_pnl += delta_pnl
        self.episode_spread_captured += spread_captured
        
        if trade_size > 0:
            fee = trade_size * midprice * self.transaction_fee
            self.episode_fees += fee
            self.episode_trades += 1
        
        # Update volatility tracking
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
        
        # Calculate volatility z-score
        if len(self.volatility_history) > 10:
            self.volatility_mean = np.mean(self.volatility_history)
            self.volatility_std = np.std(self.volatility_history) + 1e-8
            vol_z = (volatility - self.volatility_mean) / self.volatility_std
        else:
            vol_z = 0.0
        
        # Base reward: realized PnL + spread capture bonus
        reward = delta_pnl + 0.1 * spread_captured
        
        # Adaptive inventory penalties
        # λ_t = λ * (1 + α * vol_z)
        lambda_t = self.lambda_base * (1 + self.volatility_alpha * vol_z)
        
        # κ = 3 * λ, τ = 0.5 * λ
        kappa = self.kappa_multiplier * self.lambda_base
        tau = self.tau_multiplier * self.lambda_base
        
        # Penalty = λ_t * inv^2 + κ * max(0, |inv|-H)^2 + τ * max(0, |inv|-H)
        inventory_penalty = lambda_t * inventory**2
        
        over_threshold = max(0, abs(inventory) - self.H)
        soft_limit_penalty = kappa * (over_threshold ** 2) + tau * over_threshold
        
        reward -= inventory_penalty + soft_limit_penalty
        
        # Transaction fees
        if trade_size > 0:
            fee = trade_size * midprice * self.transaction_fee
            reward -= fee
        
        # Risk controls
        if abs(inventory) > self.max_inventory:
            reward -= 200.0  # Large penalty for exceeding inventory limit
        
        if self.episode_pnl < self.kill_switch_threshold:
            reward -= 2000.0  # Kill switch penalty
        
        if self.episode_pnl < -self.max_loss_per_episode:
            reward -= 1000.0  # Per-episode loss limit penalty
        
        return reward
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get enhanced episode statistics."""
        return {
            'episode_pnl': self.episode_pnl,
            'episode_fees': self.episode_fees,
            'episode_trades': self.episode_trades,
            'episode_spread_captured': self.episode_spread_captured,
            'net_pnl': self.episode_pnl - self.episode_fees,
            'lambda_base': self.lambda_base,
            'volatility_mean': self.volatility_mean,
            'volatility_std': self.volatility_std
        }
    
    def is_episode_terminated(self) -> bool:
        """Check if episode should be terminated due to risk controls."""
        return (abs(self.episode_pnl) > abs(self.kill_switch_threshold) or
                self.episode_pnl < -self.max_loss_per_episode)
