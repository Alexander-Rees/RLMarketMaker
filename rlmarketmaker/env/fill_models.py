"""Probabilistic fill models for market making."""

import numpy as np
from typing import Tuple


class ExponentialFillModel:
    """Exponential decay fill probability model."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        """
        Initialize fill model.
        
        Args:
            alpha: Base fill probability (at distance 0)
            beta: Decay rate (higher = faster decay)
        """
        self.alpha = alpha
        self.beta = beta
    
    def fill_probability(self, distance: float) -> float:
        """
        Calculate fill probability based on distance from best bid/ask.
        
        Args:
            distance: Distance from best bid/ask (in ticks or price units)
            
        Returns:
            Fill probability [0, 1]
        """
        # Exponential decay: P(fill) = alpha * exp(-beta * distance)
        prob = self.alpha * np.exp(-self.beta * distance)
        return min(prob, 1.0)  # Cap at 1.0
    
    def is_filled(self, distance: float, rng: np.random.RandomState) -> bool:
        """
        Determine if an order is filled.
        
        Args:
            distance: Distance from best bid/ask
            rng: Random number generator
            
        Returns:
            True if filled, False otherwise
        """
        prob = self.fill_probability(distance)
        return rng.random() < prob
    
    def partial_fill(self, distance: float, size: float, rng: np.random.RandomState) -> float:
        """
        Calculate partial fill size.
        
        Args:
            distance: Distance from best bid/ask
            size: Order size
            rng: Random number generator
            
        Returns:
            Filled size (0 to size)
        """
        if self.is_filled(distance, rng):
            # Full fill with some probability, otherwise partial
            if rng.random() < 0.7:  # 70% chance of full fill
                return size
            else:
                # Partial fill: 20-80% of size
                fill_ratio = rng.uniform(0.2, 0.8)
                return size * fill_ratio
        return 0.0


class LinearFillModel:
    """Linear decay fill probability model."""
    
    def __init__(self, max_distance: float = 10.0):
        """
        Initialize linear fill model.
        
        Args:
            max_distance: Distance at which fill probability reaches 0
        """
        self.max_distance = max_distance
    
    def fill_probability(self, distance: float) -> float:
        """Linear decay fill probability."""
        if distance >= self.max_distance:
            return 0.0
        return max(0.0, 1.0 - distance / self.max_distance)
    
    def is_filled(self, distance: float, rng: np.random.RandomState) -> bool:
        """Determine if an order is filled."""
        prob = self.fill_probability(distance)
        return rng.random() < prob
    
    def partial_fill(self, distance: float, size: float, rng: np.random.RandomState) -> float:
        """Calculate partial fill size."""
        if self.is_filled(distance, rng):
            # Linear decay in fill size
            fill_ratio = max(0.1, 1.0 - distance / self.max_distance)
            return size * fill_ratio
        return 0.0


def calibrate_fill_model(historical_data: np.ndarray, distances: np.ndarray) -> Tuple[float, float]:
    """
    Calibrate exponential fill model parameters from historical data.
    
    Args:
        historical_data: Array of fill rates at different distances
        distances: Array of distances corresponding to fill rates
        
    Returns:
        Tuple of (alpha, beta) parameters
    """
    # Simple linear regression on log-transformed data
    # log(P) = log(alpha) - beta * distance
    log_fill_rates = np.log(np.maximum(historical_data, 1e-6))
    
    # Linear regression: y = a + b*x where y = log(P), x = distance
    n = len(distances)
    sum_x = np.sum(distances)
    sum_y = np.sum(log_fill_rates)
    sum_xy = np.sum(distances * log_fill_rates)
    sum_x2 = np.sum(distances**2)
    
    # Solve for beta and log(alpha)
    beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    log_alpha = (sum_y - beta * sum_x) / n
    
    alpha = np.exp(log_alpha)
    return alpha, beta
