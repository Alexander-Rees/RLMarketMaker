# RL Market Maker

A reinforcement learning framework for algorithmic market making using PPO (Proximal Policy Optimization) and baseline strategies.

## Overview

This project trains a reinforcement learning agent to act as a market maker in financial markets, learning to quote bid and ask prices to:
- Earn the bid-ask spread (buy low, sell high)
- Provide liquidity to other market participants  
- Control inventory risk (avoid large long/short positions)

The goal is to maximize long-term expected profit while minimizing volatility and drawdowns.

## How It Works

### Environment
- Based on synthetic order-book data (Geometric Brownian Motion + Poisson arrivals)
- Each time step provides market state: midprice, spread, volatility, inventory, etc.
- Probabilistic fill model based on quote distance from best bid/ask
- Reward: `r = ΔPnL - λ_inv * inventory² - fees`

### Agent (RL Policy)
- Implemented with Proximal Policy Optimization (PPO) from Stable-Baselines3
- Discrete action space: [bid_offset, ask_offset, size_idx]
- Observation space: [mid_returns, vol_estimate, spread, inventory, time_remaining]
- VecNormalize for stable training

### Training Features
- **Reproducible**: Seeded random number generators
- **Config-driven**: YAML configuration files
- **Risk controls**: Inventory limits, loss caps, kill switches
- **Latency simulation**: N-step delay for realistic execution
- **Domain randomization**: Vary volatility, fees, latency per episode
- **Curriculum learning**: Start easy, ramp up difficulty

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Alexander-Rees/RLMarketMaker.git
cd RLMarketMaker

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train PPO agent
python train.py --config configs/ppo.yaml --seed 42

# Train with custom config
python train.py --config configs/synthetic.yaml --seed 123
```

### Evaluation

```bash
# Evaluate all agents
python evaluate.py --episodes 10 --seed 42

# Evaluate with trained model
python evaluate.py --model logs/models/final_model.zip --episodes 20
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_env_core.py -v
```

## Baseline Strategies

1. **Fixed Spread**: Quotes ±k ticks with inventory cap
2. **Avellaneda-Stoikov**: Optimal market making with risk aversion
3. **Inventory Mean Reversion**: Adjusts quotes based on inventory
4. **Random**: Random actions for comparison

## Project Structure

```
RLMarketMaker/
├── configs/                    # Configuration files
│   ├── synthetic.yaml          # Synthetic data parameters
│   ├── binance.yaml            # Real data replay config
│   └── ppo.yaml                # PPO hyperparameters
├── rlmarketmaker/              # Main package
│   ├── env/                    # Environment components
│   │   ├── market_env.py         # Core Gymnasium environment
│   │   ├── fill_models.py       # Probabilistic fill logic
│   │   └── reward.py            # Reward calculation
│   ├── data/                    # Data feeds
│   │   └── feeds.py             # Synthetic & real data feeds
│   ├── agents/                  # RL agents & baselines
│   │   └── baselines.py         # Baseline strategies
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration loading
│       ├── metrics.py           # Performance metrics
│       └── logging.py           # Logging utilities
├── tests/                       # Test suite
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
└── requirements.txt             # Dependencies
```

## Configuration

### Synthetic Data (`configs/synthetic.yaml`)
```yaml
mu: 0.0                    # Drift (annualized)
sigma: 0.2                 # Volatility (annualized)
initial_price: 100.0      # Starting midprice
spread_mean: 0.01          # Average spread
lambda_orders: 10.0        # Poisson rate for market orders
episode_length: 1000       # Steps per episode
```

### PPO Training (`configs/ppo.yaml`)
```yaml
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
gamma: 0.99
total_timesteps: 1000000
normalize_observations: true
```

## Metrics

- **PnL**: Cumulative profit/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Inventory Variance**: Risk exposure
- **Fill Rate**: Percentage of orders filled
- **Max Drawdown**: Largest equity decline

## Testing

The project includes comprehensive tests:
- **PnL Invariant**: `cash + inventory * midprice ≈ total PnL`
- **Reward Correctness**: `reward = ΔPnL - λ·inv² - fees`
- **Latency Delay**: Actions delayed by N steps
- **Fill Monotonicity**: Fill probability decreases with distance
- **Reset Cleanliness**: No state leakage between episodes

## Usage Examples

### Basic Training
```python
from rlmarketmaker.env.market_env import MarketMakerEnv
from rlmarketmaker.data.feeds import SyntheticFeed
from rlmarketmaker.utils.config import load_config

# Load configuration
config = load_config('configs/synthetic.yaml')

# Create environment
feed = SyntheticFeed(seed=42)
env = MarketMakerEnv(feed, config, seed=42)

# Train with PPO
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Custom Baseline
```python
from rlmarketmaker.agents.baselines import FixedSpreadStrategy

# Create fixed spread strategy
strategy = FixedSpreadStrategy(
    spread_ticks=2,
    max_inventory=100.0,
    inventory_cap=50.0
)

# Get action
action = strategy.get_action(observation, market_state)
```

## Research Applications

This framework enables research in:
- **RL for Finance**: Deep RL applications in algorithmic trading
- **Market Microstructure**: Order book dynamics and liquidity provision
- **Risk Management**: Inventory control and drawdown minimization
- **Strategy Comparison**: RL vs analytical market making

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for environment interface
- Avellaneda & Stoikov for optimal market making theory

## References

1. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
2. Schulman, J., et al. (2017). Proximal policy optimization algorithms.
3. Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). Dealing with the inventory risk.
