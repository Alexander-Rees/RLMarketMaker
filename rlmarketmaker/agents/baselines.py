"""Baseline strategies for market making."""

import numpy as np
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod


class BaselineStrategy(ABC):
    """Abstract base class for baseline strategies."""
    
    @abstractmethod
    def get_action(self, observation: np.ndarray, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Get action from baseline strategy.
        
        Args:
            observation: Current observation
            market_state: Market state information
            
        Returns:
            Action array
        """
        pass


class FixedSpreadStrategy(BaselineStrategy):
    """Fixed spread strategy with inventory cap."""
    
    def __init__(self, 
                 spread_ticks: int = 2,
                 max_inventory: float = 100.0,
                 inventory_cap: float = 50.0):
        """
        Initialize fixed spread strategy.
        
        Args:
            spread_ticks: Number of ticks to quote away from mid
            max_inventory: Maximum absolute inventory
            inventory_cap: Inventory level at which to stop quoting
        """
        self.spread_ticks = spread_ticks
        self.max_inventory = max_inventory
        self.inventory_cap = inventory_cap
    
    def get_action(self, observation: np.ndarray, market_state: Dict[str, Any]) -> np.ndarray:
        """Get fixed spread action."""
        # Extract inventory from observation (normalized)
        inventory_norm = observation[3]  # inventory / max_inventory
        inventory = inventory_norm * self.max_inventory
        
        # Check inventory cap
        if abs(inventory) > self.inventory_cap:
            # Stop quoting if inventory is too high
            return np.array([9, 9, 0])  # Far away, small size
        
        # Fixed spread quotes
        bid_offset = self.spread_ticks
        ask_offset = self.spread_ticks
        size_idx = 2  # Medium size
        
        return np.array([bid_offset, ask_offset, size_idx])


class AvellanedaStoikovStrategy(BaselineStrategy):
    """Avellaneda-Stoikov optimal market making strategy."""
    
    def __init__(self,
                 gamma: float = 0.1,
                 sigma: float = 0.2,
                 T: float = 1.0,
                 k: float = 1.0,
                 max_inventory: float = 100.0):
        """
        Initialize Avellaneda-Stoikov strategy.
        
        Args:
            gamma: Risk aversion parameter
            sigma: Volatility estimate
            T: Time horizon
            k: Market impact parameter
            max_inventory: Maximum absolute inventory
        """
        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        self.k = k
        self.max_inventory = max_inventory
    
    def get_action(self, observation: np.ndarray, market_state: Dict[str, Any]) -> np.ndarray:
        """Get Avellaneda-Stoikov optimal action."""
        # Extract state variables
        inventory_norm = observation[3]  # inventory / max_inventory
        inventory = inventory_norm * self.max_inventory
        time_remaining = observation[4]  # 0 to 1
        
        # Calculate time to maturity
        t = time_remaining * self.T
        
        # Calculate reservation price
        reservation_price = self._calculate_reservation_price(inventory, t)
        
        # Calculate optimal spread
        spread = self._calculate_optimal_spread(t)
        
        # Get current midprice and spread from market state
        midprice = market_state.get('midprice', 100.0)
        current_spread = market_state.get('spread', 0.01)
        
        # Calculate quote prices
        bid_price = reservation_price - spread / 2
        ask_price = reservation_price + spread / 2
        
        # Convert to action space
        bid_offset = self._price_to_offset(bid_price, midprice, current_spread)
        ask_offset = self._price_to_offset(ask_price, midprice, current_spread)
        
        # Adjust size based on inventory
        size_idx = self._get_size_index(inventory)
        
        return np.array([bid_offset, ask_offset, size_idx])
    
    def _calculate_reservation_price(self, inventory: float, t: float) -> float:
        """Calculate reservation price."""
        # r = S - γ * σ² * q * (T - t)
        return -self.gamma * self.sigma**2 * inventory * (self.T - t)
    
    def _calculate_optimal_spread(self, t: float) -> float:
        """Calculate optimal spread."""
        # s = γ * σ² * (T - t) + (2/γ) * log(1 + γ/k)
        spread = (self.gamma * self.sigma**2 * (self.T - t) + 
                  (2 / self.gamma) * np.log(1 + self.gamma / self.k))
        return spread
    
    def _price_to_offset(self, price: float, midprice: float, current_spread: float) -> int:
        """Convert price to offset action."""
        # Calculate distance from midprice in ticks
        distance = abs(price - midprice) / (current_spread / 2)
        
        # Convert to action space (0-9)
        offset = min(int(distance * 2), 9)
        return offset
    
    def _get_size_index(self, inventory: float) -> int:
        """Get size index based on inventory."""
        # Reduce size as inventory increases
        if abs(inventory) < self.max_inventory * 0.2:
            return 4  # Large size
        elif abs(inventory) < self.max_inventory * 0.5:
            return 3  # Medium-large size
        elif abs(inventory) < self.max_inventory * 0.8:
            return 2  # Medium size
        else:
            return 1  # Small size


class RandomStrategy(BaselineStrategy):
    """Random strategy for comparison."""
    
    def __init__(self, seed: int = None):
        """Initialize random strategy."""
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray, market_state: Dict[str, Any]) -> np.ndarray:
        """Get random action."""
        return self.rng.randint(0, 10, size=3)


class InventoryMeanReversionStrategy(BaselineStrategy):
    """Simple inventory mean reversion strategy."""
    
    def __init__(self, 
                 base_spread: int = 2,
                 inventory_sensitivity: float = 0.5,
                 max_inventory: float = 100.0):
        """
        Initialize inventory mean reversion strategy.
        
        Args:
            base_spread: Base spread in ticks
            inventory_sensitivity: How much to adjust spread based on inventory
            max_inventory: Maximum absolute inventory
        """
        self.base_spread = base_spread
        self.inventory_sensitivity = inventory_sensitivity
        self.max_inventory = max_inventory
    
    def get_action(self, observation: np.ndarray, market_state: Dict[str, Any]) -> np.ndarray:
        """Get inventory mean reversion action."""
        # Extract inventory
        inventory_norm = observation[3]  # inventory / max_inventory
        inventory = inventory_norm * self.max_inventory
        
        # Adjust spread based on inventory
        # If long inventory, quote higher ask (wider spread on ask)
        # If short inventory, quote lower bid (wider spread on bid)
        
        if inventory > 0:
            # Long inventory: wider ask spread, tighter bid spread
            bid_offset = max(0, self.base_spread - int(inventory * self.inventory_sensitivity))
            ask_offset = min(9, self.base_spread + int(inventory * self.inventory_sensitivity))
        else:
            # Short inventory: wider bid spread, tighter ask spread
            bid_offset = min(9, self.base_spread + int(abs(inventory) * self.inventory_sensitivity))
            ask_offset = max(0, self.base_spread - int(abs(inventory) * self.inventory_sensitivity))
        
        # Adjust size based on inventory
        if abs(inventory) > self.max_inventory * 0.8:
            size_idx = 0  # Small size
        elif abs(inventory) > self.max_inventory * 0.5:
            size_idx = 1  # Small-medium size
        else:
            size_idx = 2  # Medium size
        
        return np.array([bid_offset, ask_offset, size_idx])


def create_baseline_strategy(strategy_name: str, config: Dict[str, Any]) -> BaselineStrategy:
    """
    Create baseline strategy from name and config.
    
    Args:
        strategy_name: Name of strategy ('fixed_spread', 'avellaneda_stoikov', 'random', 'inventory_mean_reversion')
        config: Configuration dictionary
        
    Returns:
        Baseline strategy instance
    """
    if strategy_name == 'fixed_spread':
        return FixedSpreadStrategy(
            spread_ticks=config.get('spread_ticks', 2),
            max_inventory=config.get('max_inventory', 100.0),
            inventory_cap=config.get('inventory_cap', 50.0)
        )
    elif strategy_name == 'avellaneda_stoikov':
        return AvellanedaStoikovStrategy(
            gamma=config.get('gamma', 0.1),
            sigma=config.get('sigma', 0.2),
            T=config.get('T', 1.0),
            k=config.get('k', 1.0),
            max_inventory=config.get('max_inventory', 100.0)
        )
    elif strategy_name == 'random':
        return RandomStrategy(seed=config.get('seed', None))
    elif strategy_name == 'inventory_mean_reversion':
        return InventoryMeanReversionStrategy(
            base_spread=config.get('base_spread', 2),
            inventory_sensitivity=config.get('inventory_sensitivity', 0.5),
            max_inventory=config.get('max_inventory', 100.0)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
