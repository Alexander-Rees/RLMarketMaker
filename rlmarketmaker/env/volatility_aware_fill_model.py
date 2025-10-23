"""
Volatility-Aware Fill Model

Implements realistic fill probabilities that decrease with:
1. Distance from midprice (offset_ticks)
2. Current market volatility

Formula: p_fill = α * exp(-β * offset_ticks) * exp(-c * current_vol)

Where:
- α: Base fill probability (0.7)
- β: Distance decay rate (3.0)  
- c: Volatility penalty coefficient (0.75)
- current_vol: Current market volatility (annualized)
"""

import numpy as np
from typing import Optional


class VolatilityAwareFillModel:
    """Fill model that accounts for market volatility."""
    
    def __init__(self, 
                 alpha: float = 0.7,
                 beta: float = 3.0,
                 volatility_penalty_c: float = 0.75):
        """
        Initialize volatility-aware fill model.
        
        Args:
            alpha: Base fill probability at midprice
            beta: Distance decay rate
            volatility_penalty_c: Volatility penalty coefficient
        """
        self.alpha = alpha
        self.beta = beta
        self.volatility_penalty_c = volatility_penalty_c
    
    def fill_probability(self, 
                        offset_ticks: float, 
                        current_volatility: float = 0.0) -> float:
        """
        Calculate fill probability based on distance and volatility.
        
        Args:
            offset_ticks: Distance from midprice in ticks
            current_volatility: Current market volatility (annualized)
            
        Returns:
            Fill probability between 0 and 1
        """
        # Base probability from distance
        distance_prob = self.alpha * np.exp(-self.beta * offset_ticks)
        
        # Volatility penalty
        volatility_penalty = np.exp(-self.volatility_penalty_c * current_volatility)
        
        # Combined probability
        fill_prob = distance_prob * volatility_penalty
        
        # Ensure probability is in valid range
        return np.clip(fill_prob, 0.0, 1.0)
    
    def get_fill_probability_at_distance(self, 
                                       offset_ticks: float, 
                                       volatility: float = 0.0) -> float:
        """Get fill probability at specific distance and volatility."""
        return self.fill_probability(offset_ticks, volatility)
    
    def get_volatility_impact(self, 
                             volatility: float) -> float:
        """Get the impact of volatility on fill probability."""
        return np.exp(-self.volatility_penalty_c * volatility)
