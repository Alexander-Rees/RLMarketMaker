"""Latency and timing tests."""

import pytest
import numpy as np
from rlmarketmaker.env.market_env import MarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config


class TestLatency:
    """Test latency functionality."""
    
    def setup_method(self):
        """Set up test environment with latency."""
        config = load_config("configs/synthetic.yaml")
        config.update({
            'episode_length': 50,
            'latency': 3,  # 3-step latency
            'max_inventory': 100.0
        })
        
        feed = SyntheticFeed(seed=42)
        self.env = MarketMakerEnv(feed, config, seed=42)
    
    def test_latency_delay(self):
        """Test that actions are delayed by latency steps."""
        obs, _ = self.env.reset()
        
        # Take an action
        action = np.array([1, 1, 1])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Action should be in latency queue
        assert len(self.env.latency_queue) == 1, "Action should be in latency queue"
        assert self.env.latency_queue[0]['step'] == self.env.step_count + self.env.latency
        
        # Step forward to execute the action (latency = 3, so need 3 more steps)
        for _ in range(3):
            obs, reward, terminated, truncated, info = self.env.step(np.array([0, 0, 0]))
            if terminated or truncated:
                break
        
        # Action should be executed (queue should have 4 actions: 3 new ones + 1 more from the last step)
        assert len(self.env.latency_queue) == 4, "Should have 4 actions in queue after execution"
    
    def test_multiple_latency_actions(self):
        """Test multiple actions in latency queue."""
        obs, _ = self.env.reset()
        
        # Take multiple actions
        actions = [
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
            np.array([3, 3, 3])
        ]
        
        for action in actions:
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        
        # All actions should be in queue
        assert len(self.env.latency_queue) == 3, f"Should have 3 actions in queue, got {len(self.env.latency_queue)}"
        
        # Check execution order (actions execute in the order they were submitted)
        for i, action_data in enumerate(self.env.latency_queue):
            expected_step = self.env.step_count + self.env.latency - (2 - i)  # First action executes first
            assert action_data['step'] == expected_step, f"Action {i} should execute at step {expected_step}"
    
    def test_latency_execution_order(self):
        """Test that actions are executed in correct order."""
        obs, _ = self.env.reset()
        
        # Take actions with different timestamps
        action1 = np.array([1, 1, 1])
        obs, reward, terminated, truncated, info = self.env.step(action1)
        
        action2 = np.array([2, 2, 2])
        obs, reward, terminated, truncated, info = self.env.step(action2)
        
        # Check execution order
        assert self.env.latency_queue[0]['step'] < self.env.latency_queue[1]['step'], "Actions should execute in order"
    
    def test_latency_with_episode_end(self):
        """Test latency behavior when episode ends."""
        obs, _ = self.env.reset()
        
        # Take action near episode end
        for _ in range(self.env.episode_length - 2):
            action = np.array([1, 1, 1])
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # Episode should end with actions still in queue
        assert len(self.env.latency_queue) > 0, "Should have actions in queue when episode ends"
    
    def test_latency_reset(self):
        """Test that latency queue is reset between episodes."""
        obs, _ = self.env.reset()
        
        # Take some actions
        for _ in range(3):
            action = np.array([1, 1, 1])
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Latency queue should be empty
        assert len(self.env.latency_queue) == 0, "Latency queue should be empty after reset"
    
    def test_latency_timing(self):
        """Test that latency timing is correct."""
        obs, _ = self.env.reset()
        
        # Take action at step 0
        action = np.array([1, 1, 1])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Action should execute at step 4 (0 + latency + 1)
        assert self.env.latency_queue[0]['step'] == 4, f"Action should execute at step 4, got {self.env.latency_queue[0]['step']}"
        
        # Step forward to execution
        for _ in range(4):
            obs, reward, terminated, truncated, info = self.env.step(np.array([0, 0, 0]))
            if terminated or truncated:
                break
        
        # Action should be executed (queue should have 4 new actions, not the original one)
        assert len(self.env.latency_queue) == 4, "Should have 4 new actions in queue after execution"
    
    def test_latency_with_different_latency_values(self):
        """Test latency with different latency values."""
        for latency in [1, 2, 5, 10]:
            config = load_config("configs/synthetic.yaml")
            config.update({
                'episode_length': 20,
                'latency': latency,
                'max_inventory': 100.0
            })
            
            feed = SyntheticFeed(seed=42)
            env = MarketMakerEnv(feed, config, seed=42)
            
            obs, _ = env.reset()
            
            # Take action
            action = np.array([1, 1, 1])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check latency queue
            assert len(env.latency_queue) == 1, f"Should have 1 action in queue for latency {latency}"
            assert env.latency_queue[0]['step'] == latency + 1, f"Action should execute at step {latency + 1}"
            
            # Step forward to execution
            for _ in range(latency + 1):
                obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
                if terminated or truncated:
                    break
            
            # Action should be executed (queue should have latency+1 new actions, not the original one)
            assert len(env.latency_queue) == latency + 1, f"Should have {latency + 1} new actions in queue after execution"
