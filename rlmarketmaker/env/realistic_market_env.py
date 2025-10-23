"""
Realistic Market Making Environment with Adverse Selection and Volatility-Aware Fills

Key Realism Patches:
1. Adverse Selection Bias: After fills, apply drift opposite to fill direction
   - Formula: mid[t+1] = mid[t] + σ * ε_t + drift
   - drift = -η * np.sign(fill_side) * proximity_factor
   - η ∈ [0.0001, 0.001], proximity_factor ∈ [0.5, 1.0]

2. Volatility-Aware Fill Probability:
   - p_fill = α * exp(-β * offset_ticks) * exp(-c * current_vol)
   - c ∈ [0.5, 1.0] for high-vol penalty

3. Enhanced Inventory Costs:
   - reward -= λ * inventory**2 + κ * abs(inventory)
   - λ = 1e-3 (quadratic), κ = 1e-4 (linear)

4. Latency Enforcement:
   - All actions delayed by latency_ticks
   - Use book state at execution time, not decision time

5. Slippage/Execution Price:
   - exec_price = quote_price ± slippage
   - slippage = slippage_coeff * abs(current_return)
   - slippage_coeff ∈ [0.2, 0.5]
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import deque

from ..data.feeds import Feed, SyntheticFeed, MarketTick
from .volatility_aware_fill_model import VolatilityAwareFillModel
from .enhanced_reward import EnhancedRewardCalculator


class RealisticMarketMakerEnv(gym.Env):
    """Realistic market making environment with adverse selection and volatility-aware fills."""
    
    def __init__(self, 
                 feed: Feed,
                 config: Dict[str, Any],
                 seed: Optional[int] = None):
        """
        Initialize realistic market making environment.
        
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
        
        # Realism parameters
        self.adverse_selection_eta = config.get('adverse_selection_eta', 0.0005)  # η
        self.volatility_penalty_c = config.get('volatility_penalty_c', 0.75)      # c
        self.slippage_coeff = config.get('slippage_coeff', 0.3)                   # slippage coefficient
        
        # Action space: discrete [bid_offset, ask_offset, size_idx]
        self.action_space = gym.spaces.MultiDiscrete([10, 10, 5])
        
        # Observation space: [mid_returns, vol_est, spread, inventory, time_remaining]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # Initialize components
        self.fill_model = VolatilityAwareFillModel(
            alpha=config.get('fill_alpha', 0.7),
            beta=config.get('fill_beta', 3.0),
            volatility_penalty_c=self.volatility_penalty_c
        )
        self.reward_calc = EnhancedRewardCalculator(config)
        
        # State variables
        self.current_tick = None
        self.market_data = None
        self.step_count = 0
        self.episode_count = 0
        
        # Agent state
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        
        # Latency queue for delayed actions
        self.latency_queue = deque()
        
        # Market state tracking for adverse selection
        self.last_fill_side = None
        self.last_fill_proximity = 0.0
        self.current_volatility = 0.0
        self.price_history = deque(maxlen=20)  # For volatility calculation
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Reset state
        self.step_count = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.latency_queue.clear()
        self.price_history.clear()
        self.last_fill_side = None
        self.last_fill_proximity = 0.0
        self.current_volatility = 0.0
        
        # Reset reward calculator
        self.reward_calc.reset()
        
        # Get new market data
        self.market_data = self.feed.get_env_feed(self.config)
        self.current_tick = next(self.market_data)
        self.price_history.append(self.current_tick.midprice)
        
        self.episode_count += 1
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with realistic market dynamics."""
        # Process delayed actions from latency queue
        self._process_latency_queue()
        
        # Add current action to latency queue
        self.latency_queue.append({
            'action': action,
            'step': self.step_count + self.latency + 1
        })
        
        # Get next market tick with adverse selection
        try:
            self.current_tick = next(self.market_data)
            
            # Apply adverse selection if we had a fill last step
            if self.last_fill_side is not None:
                self._apply_adverse_selection()
            
        except StopIteration:
            # Episode finished
            return self._get_observation(), 0.0, True, False, {}
        
        # Update volatility estimate
        self._update_volatility()
        
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
            'volatility': self.current_volatility,
            'last_fill_side': self.last_fill_side,
            'midprice': self.current_tick.midprice,
            'bid_quote': bid_quote,
            'ask_quote': ask_quote,
            'filled_bid': filled_bid,
            'filled_ask': filled_ask
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_adverse_selection(self):
        """Apply adverse selection bias to market price."""
        if self.last_fill_side is None:
            return
        
        # Calculate drift based on fill side and proximity
        proximity_factor = 1.0 - self.last_fill_proximity  # Closer to mid = higher impact
        drift = -self.adverse_selection_eta * np.sign(self.last_fill_side) * proximity_factor
        
        # Apply drift to current midprice
        self.current_tick.midprice += drift
        
        # Reset for next step
        self.last_fill_side = None
        self.last_fill_proximity = 0.0
    
    def _update_volatility(self):
        """Update current volatility estimate."""
        self.price_history.append(self.current_tick.midprice)
        
        if len(self.price_history) >= 10:
            # Calculate rolling volatility
            returns = np.diff(np.log(self.price_history))
            self.current_volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
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
        """Execute a market making action with realistic fills and slippage."""
        bid_offset, ask_offset, size_idx = action
        
        # Convert action indices to actual values
        bid_offset_ticks = bid_offset * 0.5  # 0-4.5 ticks
        ask_offset_ticks = ask_offset * 0.5  # 0-4.5 ticks
        size = (size_idx + 1) * 10.0  # 10-50 units
        
        # Calculate quote prices
        bid_price = self.current_tick.midprice - self.current_tick.spread/2 - bid_offset_ticks
        ask_price = self.current_tick.midprice + self.current_tick.spread/2 + ask_offset_ticks
        
        # Check for fills with volatility-aware probability
        bid_distance = max(0, bid_offset_ticks)
        ask_distance = max(0, ask_offset_ticks)
        
        # Process bid fill
        if bid_distance == 0:  # At touch
            fill_prob = 0.8  # High probability at touch
        else:
            fill_prob = self.fill_model.fill_probability(bid_distance, self.current_volatility)
        
        if self.rng.random() < fill_prob:
            # Apply slippage to execution price
            slippage = self.slippage_coeff * abs(self.current_volatility) * self.rng.normal(0, 1)
            exec_price = bid_price - slippage
            self._process_fill('bid', exec_price, size)
            self.last_fill_side = 1  # Positive for bid fill
            self.last_fill_proximity = bid_distance / 5.0  # Normalize to [0, 1]
        
        # Process ask fill
        if ask_distance == 0:  # At touch
            fill_prob = 0.8  # High probability at touch
        else:
            fill_prob = self.fill_model.fill_probability(ask_distance, self.current_volatility)
        
        if self.rng.random() < fill_prob:
            # Apply slippage to execution price
            slippage = self.slippage_coeff * abs(self.current_volatility) * self.rng.normal(0, 1)
            exec_price = ask_price + slippage
            self._process_fill('ask', exec_price, size)
            self.last_fill_side = -1  # Negative for ask fill
            self.last_fill_proximity = ask_distance / 5.0  # Normalize to [0, 1]
    
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
        """Calculate reward with enhanced inventory costs."""
        # Calculate PnL change
        current_value = self.inventory * self.current_tick.midprice
        total_value = self.cash + current_value
        delta_pnl = total_value - self.total_pnl
        self.total_pnl = total_value
        
        # Calculate reward with enhanced inventory penalties
        reward = self.reward_calc.calculate_reward(
            delta_pnl=delta_pnl,
            inventory=self.inventory,
            trade_size=0.0,
            midprice=self.current_tick.midprice,
            volatility=self.current_volatility
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
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Calculate midprice returns
        if len(self.price_history) > 1:
            mid_returns = (self.current_tick.midprice - self.price_history[-2]) / self.price_history[-2]
        else:
            mid_returns = 0.0
        
        # Volatility estimate
        vol_est = self.current_volatility
        
        # Spread
        spread = self.current_tick.spread
        
        # Inventory (normalized)
        inventory_norm = self.inventory / self.max_inventory
        
        # Time remaining
        time_remaining = 1.0 - (self.step_count / self.episode_length)
        
        return np.array([mid_returns, vol_est, spread, inventory_norm, time_remaining], dtype=np.float32)
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.step_count}, "
                  f"Inventory: {self.inventory:.2f}, "
                  f"Cash: {self.cash:.2f}, "
                  f"PnL: {self.total_pnl:.2f}, "
                  f"Mid: {self.current_tick.midprice:.2f}, "
                  f"Vol: {self.current_volatility:.4f}")
    
    def close(self):
        """Clean up environment."""
        pass
