"""
Minimal PPO implementation for RL Market Maker.
Lightweight, library-free PPO trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
import pickle
import os


class MLPPolicy(nn.Module):
    """Minimal MLP policy network."""
    
    def __init__(self, state_dim: int, action_dims: List[int], hidden_size: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dims = action_dims
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        
        # Actor heads for each action dimension
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_size, dim) for dim in action_dims
        ])
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        """Forward pass through network."""
        shared_features = self.shared(state)
        
        # Actor outputs
        action_logits = []
        for head in self.actor_heads:
            logits = head(shared_features)
            action_logits.append(logits)
        
        # Critic output
        value = self.critic(shared_features)
        
        return action_logits, value
    
    def get_action(self, state, deterministic=False):
        """Get action from policy."""
        with torch.no_grad():
            action_logits, value = self.forward(state)
            
            if deterministic:
                actions = []
                for logits in action_logits:
                    action = torch.argmax(logits, dim=-1)
                    actions.append(action)
                return torch.stack(actions, dim=-1), value
            
            # Sample actions
            actions = []
            log_probs = []
            for logits in action_logits:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action)
                log_probs.append(log_prob)
            
            return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1), value


class VecNormalizeLite:
    """Lightweight observation normalization."""
    
    def __init__(self, obs_dim: int, epsilon: float = 1e-8):
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.running_mean = np.zeros(obs_dim, dtype=np.float32)
        self.running_var = np.ones(obs_dim, dtype=np.float32)
        self.count = 0
        
    def update(self, obs):
        """Update running statistics."""
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean += delta * batch_count / total_count
        self.running_var += (batch_var - self.running_var) * batch_count / total_count
        self.count = total_count
    
    def normalize(self, obs):
        """Normalize observations."""
        if self.count == 0:
            return obs
        
        return (obs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
    
    def save(self, path: str):
        """Save normalization stats."""
        stats = {
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'count': self.count
        }
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
    
    def load(self, path: str):
        """Load normalization stats."""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        self.running_mean = stats['running_mean']
        self.running_var = stats['running_var']
        self.count = stats['count']


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dims: List[int]):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dims = action_dims
        
        # Storage
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, len(action_dims)), dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.log_probs = np.zeros((buffer_size, len(action_dims)), dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, done, log_prob, value):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, gamma: float, gae_lambda: float, next_value: float = 0.0):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros(self.size, dtype=np.float32)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        self.advantages = advantages
        self.returns = advantages + self.values[:self.size]
    
    def get_batch(self, batch_size: int):
        """Get random batch from buffer."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


class MinPPO:
    """Minimal PPO trainer."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dims: List[int],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = "cpu"):
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.device = device
        
        # Create policy
        self.policy = MLPPolicy(state_dim, action_dims).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Normalization
        self.vec_normalize = VecNormalizeLite(state_dim)
        
    def get_action(self, state, deterministic=False):
        """Get action from policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if deterministic:
            action, value = self.policy.get_action(state_tensor, deterministic=True)
            return action.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()
        else:
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic=False)
            return (action.squeeze(0).cpu().numpy(), 
                    log_prob.squeeze(0).cpu().numpy(), 
                    value.squeeze(0).cpu().numpy())
    
    def update(self, buffer: RolloutBuffer, epochs: int = 4, batch_size: int = 64):
        """Update policy using PPO."""
        # Compute GAE
        buffer.compute_gae(self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = buffer.advantages[:buffer.size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages[:buffer.size] = advantages
        
        # Training loop
        for epoch in range(epochs):
            for _ in range(buffer.size // batch_size):
                batch = buffer.get_batch(batch_size)
                
                # Convert to tensors
                states = torch.tensor(batch['states'], dtype=torch.float32).to(self.device)
                actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
                old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32).to(self.device)
                advantages = torch.tensor(batch['advantages'], dtype=torch.float32).to(self.device)
                returns = torch.tensor(batch['returns'], dtype=torch.float32).to(self.device)
                
                # Forward pass
                action_logits, values = self.policy(states)
                
                # Compute new log probabilities
                new_log_probs = []
                for i, (action, logits) in enumerate(zip(actions.T, action_logits)):
                    log_probs = torch.log_softmax(logits, dim=-1)
                    new_log_probs.append(log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1))
                new_log_probs = torch.stack(new_log_probs, dim=-1)
                
                # Compute ratios
                old_log_probs_sum = old_log_probs.sum(dim=-1)
                new_log_probs_sum = new_log_probs.sum(dim=-1)
                ratios = torch.exp(new_log_probs_sum - old_log_probs_sum)
                
                # Compute losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values.squeeze(), returns)
                
                # Entropy loss
                entropy_loss = 0
                for logits in action_logits:
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    entropy_loss += entropy
                
                total_loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path: str):
        """Save model and normalization stats."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save policy
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{path}.pt")
        
        # Save normalization
        self.vec_normalize.save(f"{path}_vecnorm.pkl")
    
    def load(self, path: str):
        """Load model and normalization stats."""
        # Load policy
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load normalization
        self.vec_normalize.load(f"{path}_vecnorm.pkl")
