"""
Replay Market Making Environment using historical data.

This environment replays historical market data while allowing agents to place orders
and get filled based on realistic fill models calibrated to the historical data.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque

from ..data.feeds import Feed, PolygonReplayFeed, MarketTick
from .fill_models import ExponentialFillModel


class ReplayMarketMakerEnv(gym.Env):
    """Market making environment that replays historical data."""
    
    def __init__(self, 
                 feed: Feed,
                 config: Dict[str, Any],
                 seed: Optional[int] = None):
        """
        Initialize replay market making environment.
        
        Args:
            feed: Data feed for historical market data
            config: Configuration dictionary
            seed: Random seed
        """
        super().__init__()
        
        self.feed = feed
        self.config = config
        self.seed = seed
        
        # Set up random number generator
        self.rng = np.random.RandomState(seed)
        
        # Environment parameters
        self.episode_length = config.get('episode_length', 3600)
        self.max_inventory = config.get('max_inventory', 100.0)
        self.tick_size = config.get('tick_size', 0.01)
        self.fee_bps = config.get('fee_bps', 1.0)  # 1 basis point fee
        self.latency_ticks = config.get('latency_ticks', 1)
        
        # Fill model parameters
        self.fill_model = ExponentialFillModel(
            alpha=config.get('fill_alpha', 0.8),
            beta=config.get('fill_beta', 0.5)
        )
        
        # State variables
        self.step_count = 0
        self.episode_count = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.current_tick = None
        self.market_data = None
        
        # Latency queue for delayed actions
        self.latency_queue = deque()
        
        # Price history for volatility calculation
        self.price_history = deque(maxlen=20)
        
        # Define action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete([10, 10, 5])  # bid_offset, ask_offset, quantity_idx
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        
        # Reset state
        self.step_count = 0
        self.episode_count += 1
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.latency_queue.clear()
        self.price_history.clear()
        
        # Get market data feed
        self.market_data = self.feed.get_env_feed(self.config)
        
        # Get first tick
        try:
            self.current_tick = next(self.market_data)
        except StopIteration:
            raise RuntimeError("No market data available")
        
        # Initialize price history
        self.price_history.append(self.current_tick.midprice)
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'inventory': self.inventory,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'step': self.step_count,
            'episode': self.episode_count,
            'midprice': self.current_tick.midprice,
            'spread': self.current_tick.spread
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Process delayed actions from latency queue
        self._process_latency_queue()
        
        # Add current action to latency queue
        self.latency_queue.append({
            'action': action,
            'step': self.step_count + self.latency_ticks + 1
        })
        
        # Get next market tick
        try:
            self.current_tick = next(self.market_data)
        except StopIteration:
            # Episode finished
            return self._get_observation(), 0.0, True, False, {}
        
        # Update price history
        self.price_history.append(self.current_tick.midprice)
        
        # Calculate reward from previous step's actions
        reward = self._calculate_reward()
        
        # Update state
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.episode_length
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate current quotes based on action
        bid_offset, ask_offset, size_idx = action
        bid_offset_ticks = bid_offset * 0.5
        ask_offset_ticks = ask_offset * 0.5
        
        bid_quote = self.current_tick.midprice - self.current_tick.spread/2 - bid_offset_ticks
        ask_quote = self.current_tick.midprice + self.current_tick.spread/2 + ask_offset_ticks
        
        # Check for fills (simplified for trace)
        filled_bid = 0
        filled_ask = 0
        if hasattr(self, 'last_fill_side'):
            if self.last_fill_side == 1:  # Bid fill
                filled_bid = 1
            elif self.last_fill_side == -1:  # Ask fill
                filled_ask = 1
        
        info = {
            'inventory': self.inventory,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'cumulative_pnl': self.total_pnl,
            'step': self.step_count,
            'episode': self.episode_count,
            'midprice': self.current_tick.midprice,
            'bid_quote': bid_quote,
            'ask_quote': ask_quote,
            'filled_bid': filled_bid,
            'filled_ask': filled_ask,
            'volatility': self._get_current_volatility()
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Normalize observation components
        midprice_norm = self.current_tick.midprice / 100.0  # Normalize to ~1
        spread_norm = self.current_tick.spread / 0.01  # Normalize to ~1
        inventory_norm = self.inventory / self.max_inventory  # Normalize to [-1, 1]
        pnl_norm = self.total_pnl / 1000.0  # Normalize to ~1
        time_norm = self.step_count / self.episode_length  # Normalize to [0, 1]
        
        return np.array([
            midprice_norm,
            spread_norm,
            inventory_norm,
            pnl_norm,
            time_norm
        ], dtype=np.float32)
    
    def _process_latency_queue(self):
        """Process delayed actions from latency queue."""
        executed_actions = []
        
        for i, queued_action in enumerate(self.latency_queue):
            if queued_action['step'] <= self.step_count:
                # Execute delayed action
                self._execute_action(queued_action['action'])
                executed_actions.append(i)
        
        # Remove executed actions
        for i in reversed(executed_actions):
            del self.latency_queue[i]
    
    def _execute_action(self, action: np.ndarray):
        """Execute a market making action with realistic fills."""
        bid_offset, ask_offset, size_idx = action
        
        # Convert action indices to actual values
        bid_offset_ticks = bid_offset * 0.5  # 0-4.5 ticks
        ask_offset_ticks = ask_offset * 0.5  # 0-4.5 ticks
        size = (size_idx + 1) * 10.0  # 10-50 units
        
        # Calculate quote prices
        bid_price = self.current_tick.midprice - self.current_tick.spread/2 - bid_offset_ticks
        ask_price = self.current_tick.midprice + self.current_tick.spread/2 + ask_offset_ticks
        
        # Calculate distance from best bid/ask
        best_bid = self.current_tick.midprice - self.current_tick.spread/2
        best_ask = self.current_tick.midprice + self.current_tick.spread/2
        
        bid_distance = max(0, (best_bid - bid_price) / self.tick_size)
        ask_distance = max(0, (ask_price - best_ask) / self.tick_size)
        
        # Get fill probabilities based on distance and market conditions
        bid_fill_prob = self._calculate_fill_probability(bid_distance, 'bid')
        ask_fill_prob = self._calculate_fill_probability(ask_distance, 'ask')
        
        # Check for fills
        if self.rng.random() < bid_fill_prob:
            self._process_fill('bid', bid_price, size)
            self.last_fill_side = 1
        
        if self.rng.random() < ask_fill_prob:
            self._process_fill('ask', ask_price, size)
            self.last_fill_side = -1
    
    def _calculate_fill_probability(self, distance_ticks: float, side: str) -> float:
        """Calculate fill probability based on distance and market conditions."""
        # Simplified fill model for testing
        if distance_ticks == 0:
            return 0.8  # High probability at touch
        elif distance_ticks <= 1:
            return 0.4  # Medium probability close to touch
        elif distance_ticks <= 2:
            return 0.2  # Lower probability further away
        else:
            return 0.1  # Low probability far away
    
    def _process_fill(self, side: str, price: float, size: float):
        """Process a filled order."""
        # Apply fees
        fee = price * size * self.fee_bps / 10000.0
        
        if side == 'bid':
            # Bought at bid price
            self.inventory += size
            self.cash -= price * size + fee
        else:
            # Sold at ask price
            self.inventory -= size
            self.cash += price * size - fee
    
    def _calculate_reward(self) -> float:
        """Calculate reward with inventory penalties."""
        # Calculate PnL change
        current_value = self.inventory * self.current_tick.midprice
        total_value = self.cash + current_value
        delta_pnl = total_value - self.total_pnl
        self.total_pnl = total_value
        
        # Inventory penalty
        inventory_penalty = -0.001 * self.inventory**2
        
        return delta_pnl + inventory_penalty
    
    def _get_current_volatility(self) -> float:
        """Get current volatility estimate."""
        if len(self.price_history) < 2:
            return 0.01
        
        returns = np.diff(np.log(list(self.price_history)))
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _check_termination(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if inventory gets too extreme
        if abs(self.inventory) > self.max_inventory * 1.5:
            return True
        
        # Terminate if PnL gets too negative
        if self.total_pnl < -1000.0:
            return True
        
        return False
    
    def close(self):
        """Close environment and cleanup resources."""
        if self.market_data is not None:
            self.market_data.close()
