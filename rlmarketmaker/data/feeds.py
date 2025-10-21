"""Data feeds for market making environment."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MarketTick:
    """Single market tick data."""
    timestamp: float
    midprice: float
    spread: float
    bid_size: float
    ask_size: float
    trades: int  # Number of trades in this tick


class Feed(ABC):
    """Abstract base class for market data feeds."""
    
    @abstractmethod
    def get_env_feed(self, config: Dict[str, Any]) -> Iterator[MarketTick]:
        """Get iterator of market ticks for environment."""
        pass


class SyntheticFeed(Feed):
    """Synthetic market data feed using GBM + Poisson arrivals."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def get_env_feed(self, config: Dict[str, Any]) -> Iterator[MarketTick]:
        """Generate synthetic market data."""
        # Extract parameters
        mu = config.get('mu', 0.0)
        sigma = config.get('sigma', 0.2)
        initial_price = config.get('initial_price', 100.0)
        spread_mean = config.get('spread_mean', 0.01)
        spread_vol = config.get('spread_vol', 0.005)
        lambda_orders = config.get('lambda_orders', 10.0)
        episode_length = config.get('episode_length', 1000)
        dt = config.get('dt', 0.001)
        
        # Domain randomization
        vol_range = config.get('volatility_range', [0.15, 0.25])
        spread_range = config.get('spread_range', [0.005, 0.015])
        
        # Randomize parameters for this episode
        sigma = self.rng.uniform(*vol_range)
        spread_mean = self.rng.uniform(*spread_range)
        
        # Initialize state
        price = initial_price
        spread = spread_mean
        
        for t in range(episode_length):
            # Update midprice using GBM
            dW = self.rng.normal(0, np.sqrt(dt))
            price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            
            # Update spread (mean-reverting with volatility)
            spread_target = spread_mean + spread_vol * sigma
            spread = 0.9 * spread + 0.1 * spread_target + self.rng.normal(0, spread_vol * dt)
            spread = max(spread, 0.001)  # Minimum spread
            
            # Generate market orders (Poisson process)
            n_trades = self.rng.poisson(lambda_orders * dt)
            
            # Generate bid/ask sizes (random)
            bid_size = self.rng.exponential(100.0)
            ask_size = self.rng.exponential(100.0)
            
            yield MarketTick(
                timestamp=t * dt,
                midprice=price,
                spread=spread,
                bid_size=bid_size,
                ask_size=ask_size,
                trades=n_trades
            )


class BinanceReplayFeed(Feed):
    """Replay feed for historical Binance data."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def get_env_feed(self, config: Dict[str, Any]) -> Iterator[MarketTick]:
        """Replay historical data with domain randomization."""
        data_path = config.get('data_path', 'data/binance_btcusdt.parquet')
        episode_length = config.get('episode_length', 3600)
        warmup_steps = config.get('warmup_steps', 100)
        
        # Load data
        try:
            df = pd.read_parquet(data_path)
        except FileNotFoundError:
            # Fallback to synthetic if data not available
            print(f"Data file {data_path} not found, falling back to synthetic data")
            synthetic_feed = SyntheticFeed(seed=self.rng.randint(0, 2**32))
            yield from synthetic_feed.get_env_feed(config)
            return
        
        # Domain randomization
        vol_mult = self.rng.uniform(*config.get('volatility_multiplier', [0.8, 1.2]))
        spread_mult = self.rng.uniform(*config.get('spread_multiplier', [0.5, 1.5]))
        
        # Select random starting point
        start_idx = self.rng.randint(warmup_steps, len(df) - episode_length)
        
        # Replay data
        for i in range(episode_length):
            row = df.iloc[start_idx + i]
            
            # Apply domain randomization
            midprice = row['midprice'] * vol_mult
            spread = row['spread'] * spread_mult
            
            yield MarketTick(
                timestamp=row['timestamp'],
                midprice=midprice,
                spread=spread,
                bid_size=row.get('bid_size', 100.0),
                ask_size=row.get('ask_size', 100.0),
                trades=row.get('trades', 0)
            )
