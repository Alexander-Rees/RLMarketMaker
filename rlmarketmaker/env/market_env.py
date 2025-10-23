"""Market making environment for reinforcement learning."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import deque

from ..data.feeds import Feed, SyntheticFeed, MarketTick
from .fill_models import ExponentialFillModel
from .reward import RewardCalculator


class MarketMakerEnv(gym.Env):
    """Market making environment with discrete actions and latency."""
    
    def __init__(self, 
                 feed: Feed,
                 config: Dict[str, Any],
                 seed: Optional[int] = None):
        """
        Initialize market making environment.
        
        Args:
            feed: Data feed for market data
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.feed = feed
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Environment parameters
        self.episode_length = config.get('episode_length', 1000)
        self.latency = config.get('latency', 1)
        self.max_inventory = config.get('max_inventory', 1000.0)
        self.max_loss = config.get('max_loss_per_episode', 1000.0)
        
        # Action space: discrete [bid_offset, ask_offset, size_idx]
        # bid_offset: 0-9 (0=at touch, 9=far away)
        # ask_offset: 0-9 (0=at touch, 9=far away)  
        # size_idx: 0-4 (0=small, 4=large)
        self.action_space = gym.spaces.MultiDiscrete([10, 10, 5])
        
        # Observation space: [mid_returns, vol_est, spread, inventory, time_remaining]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # Initialize components
        self.fill_model = ExponentialFillModel(
            alpha=config.get('fill_alpha', 1.0),
            beta=config.get('fill_beta', 2.0)
        )
        self.reward_calc = RewardCalculator(config)
        
        # State variables
        self.current_tick = None
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.step_count = 0
        self.episode_count = 0
        
        # Latency queue for delayed actions
        self.latency_queue = deque()
        
        # Observation history for normalization
        self.returns_window = deque(maxlen=20)
        self.vol_estimator = deque(maxlen=50)
        
        # Market data iterator
        self.market_data = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Reset state
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.step_count = 0
        self.latency_queue.clear()
        self.returns_window.clear()
        self.vol_estimator.clear()
        
        # Reset reward calculator
        self.reward_calc.reset()
        
        # Get new market data
        self.market_data = iter(self.feed.get_env_feed(self.config))
        self.current_tick = next(self.market_data)
        
        # Initialize observation
        obs = self._get_observation()
        
        self.episode_count += 1
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Process delayed actions from latency queue
        self._process_latency_queue()
        
        # Add current action to latency queue
        self.latency_queue.append({
            'action': action,
            'step': self.step_count + self.latency + 1
        })
        
        # Get next market tick
        try:
            self.current_tick = next(self.market_data)
        except StopIteration:
            # Episode finished
            return self._get_observation(), 0.0, True, False, {}
        
        # Calculate reward from previous step's actions
        reward = self._calculate_reward()
        
        # Update state
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.episode_length
        
        # Get new observation
        obs = self._get_observation()
        
        info = {
            'inventory': self.inventory,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'step': self.step_count,
            'episode': self.episode_count
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_tick is None:
            return np.zeros(5, dtype=np.float32)
        
        # Calculate returns
        if len(self.returns_window) > 0:
            current_return = (self.current_tick.midprice - self.returns_window[-1]) / self.returns_window[-1]
        else:
            current_return = 0.0
        
        self.returns_window.append(self.current_tick.midprice)
        
        # Calculate volatility estimate
        if len(self.returns_window) >= 2:
            returns = np.array(self.returns_window)
            vol_est = np.std(np.diff(returns)) if len(returns) > 1 else 0.0
        else:
            vol_est = 0.0
        
        self.vol_estimator.append(vol_est)
        
        # Normalize spread
        spread_norm = self.current_tick.spread / self.current_tick.midprice
        
        # Normalize inventory
        inv_norm = self.inventory / self.max_inventory if self.max_inventory > 0 else 0.0
        
        # Time remaining
        time_remaining = (self.episode_length - self.step_count) / self.episode_length
        
        obs = np.array([
            current_return,
            vol_est,
            spread_norm,
            inv_norm,
            time_remaining
        ], dtype=np.float32)
        
        return obs
    
    def _process_latency_queue(self):
        """Process actions that are ready to execute."""
        executed_actions = []
        
        for i, action_data in enumerate(self.latency_queue):
            if action_data['step'] <= self.step_count:
                self._execute_action(action_data['action'])
                executed_actions.append(i)
        
        # Remove executed actions (in reverse order to maintain indices)
        for i in reversed(executed_actions):
            del self.latency_queue[i]
    
    def _execute_action(self, action: np.ndarray):
        """Execute a market making action."""
        bid_offset, ask_offset, size_idx = action
        
        # Convert action indices to actual values
        bid_offset_ticks = bid_offset * 0.5  # 0-4.5 ticks
        ask_offset_ticks = ask_offset * 0.5  # 0-4.5 ticks
        size = (size_idx + 1) * 10.0  # 10-50 units
        
        # Calculate quote prices
        bid_price = self.current_tick.midprice - self.current_tick.spread/2 - bid_offset_ticks
        ask_price = self.current_tick.midprice + self.current_tick.spread/2 + ask_offset_ticks
        
        # Check for fills
        bid_distance = max(0, bid_offset_ticks)
        ask_distance = max(0, ask_offset_ticks)
        
        # Process bid fill
        if bid_distance == 0:  # At touch
            fill_prob = 0.8  # High probability at touch
        else:
            fill_prob = self.fill_model.fill_probability(bid_distance)
        
        if self.rng.random() < fill_prob:
            self._process_fill('bid', bid_price, size)
        
        # Process ask fill
        if ask_distance == 0:  # At touch
            fill_prob = 0.8  # High probability at touch
        else:
            fill_prob = self.fill_model.fill_probability(ask_distance)
        
        if self.rng.random() < fill_prob:
            self._process_fill('ask', ask_price, size)
    
    def _process_fill(self, side: str, price: float, size: float):
        """Process a filled order."""
        if side == 'bid':
            # Bought at bid price
            self.inventory += size
            self.cash -= price * size
        else:
            # Sold at ask price
            self.inventory -= size
            self.cash += price * size
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current step."""
        # Calculate PnL change
        current_value = self.inventory * self.current_tick.midprice
        total_value = self.cash + current_value
        delta_pnl = total_value - self.total_pnl
        self.total_pnl = total_value
        
        # Calculate reward
        reward = self.reward_calc.calculate_reward(
            delta_pnl=delta_pnl,
            inventory=self.inventory,
            trade_size=0.0,  # Will be updated by reward calculator
            midprice=self.current_tick.midprice
        )
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should be terminated."""
        # Check inventory limits
        if abs(self.inventory) > self.max_inventory:
            return True
        
        # Check loss limits
        if self.total_pnl < -self.max_loss:
            return True
        
        # Check kill switch
        if self.total_pnl < self.config.get('kill_switch_threshold', -5000.0):
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.step_count}, "
                  f"Inventory: {self.inventory:.2f}, "
                  f"Cash: {self.cash:.2f}, "
                  f"PnL: {self.total_pnl:.2f}, "
                  f"Mid: {self.current_tick.midprice:.2f}")
    
    def close(self):
        """Clean up environment."""
        pass
