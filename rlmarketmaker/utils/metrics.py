"""Metrics calculation for market making performance."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


class MarketMakingMetrics:
    """Calculate performance metrics for market making strategies."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episode_pnls = []
        self.episode_returns = []
        self.inventory_history = []
        self.fill_rates = []
        self.max_drawdowns = []
        self.sharpe_ratios = []
        self.episode_lengths = []
    
    def add_episode(self, 
                   episode_pnl: float,
                   inventory_history: List[float],
                   fill_rate: float,
                   episode_length: int):
        """
        Add episode data to metrics.
        
        Args:
            episode_pnl: Total PnL for the episode
            inventory_history: List of inventory values over time
            fill_rate: Fraction of orders that were filled
            episode_length: Number of steps in episode
        """
        self.episode_pnls.append(episode_pnl)
        self.episode_returns.append(episode_pnl / episode_length if episode_length > 0 else 0)
        self.inventory_history.extend(inventory_history)
        self.fill_rates.append(fill_rate)
        self.episode_lengths.append(episode_length)
        
        # Calculate episode-specific metrics
        if len(inventory_history) > 1:
            inv_var = np.var(inventory_history)
            self.max_drawdowns.append(self._calculate_max_drawdown(inventory_history))
        else:
            self.max_drawdowns.append(0.0)
    
    def calculate_summary_metrics(self) -> Dict[str, float]:
        """Calculate summary metrics across all episodes."""
        if not self.episode_pnls:
            return {}
        
        metrics = {}
        
        # PnL metrics
        metrics['total_pnl'] = sum(self.episode_pnls)
        metrics['mean_episode_pnl'] = np.mean(self.episode_pnls)
        metrics['std_episode_pnl'] = np.std(self.episode_pnls)
        metrics['min_episode_pnl'] = np.min(self.episode_pnls)
        metrics['max_episode_pnl'] = np.max(self.episode_pnls)
        
        # Return metrics
        if len(self.episode_returns) > 1:
            metrics['mean_return'] = np.mean(self.episode_returns)
            metrics['std_return'] = np.std(self.episode_returns)
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['std_return'] if metrics['std_return'] > 0 else 0
        else:
            metrics['mean_return'] = self.episode_returns[0] if self.episode_returns else 0
            metrics['std_return'] = 0
            metrics['sharpe_ratio'] = 0
        
        # Inventory metrics
        if self.inventory_history:
            metrics['mean_inventory'] = np.mean(self.inventory_history)
            metrics['std_inventory'] = np.std(self.inventory_history)
            metrics['max_inventory'] = np.max(np.abs(self.inventory_history))
            metrics['inventory_variance'] = np.var(self.inventory_history)
        else:
            metrics['mean_inventory'] = 0
            metrics['std_inventory'] = 0
            metrics['max_inventory'] = 0
            metrics['inventory_variance'] = 0
        
        # Fill rate metrics
        metrics['mean_fill_rate'] = np.mean(self.fill_rates)
        metrics['std_fill_rate'] = np.std(self.fill_rates)
        
        # Drawdown metrics
        metrics['mean_max_drawdown'] = np.mean(self.max_drawdowns)
        metrics['max_drawdown'] = np.max(self.max_drawdowns)
        
        # Episode length metrics
        metrics['mean_episode_length'] = np.mean(self.episode_lengths)
        metrics['total_steps'] = sum(self.episode_lengths)
        
        return metrics
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from a series of values."""
        if not values:
            return 0.0
        
        values = np.array(values)
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def get_episode_metrics(self, episode_idx: int) -> Dict[str, float]:
        """Get metrics for a specific episode."""
        if episode_idx >= len(self.episode_pnls):
            return {}
        
        return {
            'episode_pnl': self.episode_pnls[episode_idx],
            'episode_return': self.episode_returns[episode_idx],
            'fill_rate': self.fill_rates[episode_idx],
            'max_drawdown': self.max_drawdowns[episode_idx],
            'episode_length': self.episode_lengths[episode_idx]
        }


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns."""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(prices: List[float]) -> float:
    """Calculate maximum drawdown from price series."""
    if not prices:
        return 0.0
    
    prices = np.array(prices)
    peak = prices[0]
    max_dd = 0.0
    
    for price in prices:
        if price > peak:
            peak = price
        dd = peak - price
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def calculate_inventory_metrics(inventory_history: List[float]) -> Dict[str, float]:
    """Calculate inventory-related metrics."""
    if not inventory_history:
        return {}
    
    inventory = np.array(inventory_history)
    
    return {
        'mean_inventory': np.mean(inventory),
        'std_inventory': np.std(inventory),
        'max_inventory': np.max(inventory),
        'min_inventory': np.min(inventory),
        'inventory_variance': np.var(inventory),
        'inventory_range': np.max(inventory) - np.min(inventory)
    }


def calculate_fill_metrics(fill_events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate fill-related metrics."""
    if not fill_events:
        return {}
    
    total_orders = len(fill_events)
    filled_orders = sum(1 for event in fill_events if event.get('filled', False))
    
    return {
        'total_orders': total_orders,
        'filled_orders': filled_orders,
        'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
        'mean_fill_size': np.mean([event.get('size', 0) for event in fill_events if event.get('filled', False)]),
        'mean_fill_price': np.mean([event.get('price', 0) for event in fill_events if event.get('filled', False)])
    }
