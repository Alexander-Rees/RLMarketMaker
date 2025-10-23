# RLMarketMaker

A reinforcement learning framework for algorithmic market making with realistic market simulation and historical data replay.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Alexander-Rees/RLMarketMaker.git
cd RLMarketMaker

# Create virtual environment
make venv
source venv/bin/activate

# Install dependencies
make install
```

### Training

```bash
# Train PPO agent on synthetic data
make train

# Or manually:
python scripts/training/train_min.py --config configs/realistic_environment.yaml --seed 42
```

### Evaluation

```bash
# Evaluate on synthetic data
python scripts/evaluation/eval_min.py --checkpoint logs/checkpoints/policy --episodes 10

# Evaluate on historical replay data
python scripts/evaluation/evaluate_replay.py --config configs/polygon_replay.yaml --checkpoint logs/checkpoints/policy.pt --episodes 10
```

### Analysis

```bash
# Generate agent behavior traces
python scripts/analysis/trace_eval.py --agent ppo --ckpt logs/checkpoints/policy.pt --steps 1000 --seed 123

# Create visualizations
python scripts/analysis/plot_traces.py
```

## 📁 Project Structure

```
RLMarketMaker/
├── rlmarketmaker/           # Core package
│   ├── agents/              # RL agents and baselines
│   │   ├── min_ppo.py      # Custom PPO implementation
│   │   └── baselines.py    # Baseline strategies
│   ├── data/               # Data feeds and preprocessing
│   │   ├── feeds.py        # Market data feeds
│   │   └── preprocess_polygon.py  # Historical data preprocessing
│   ├── env/                # Market environments
│   │   ├── realistic_market_env.py  # Realistic simulation
│   │   ├── replay_market_env.py     # Historical replay
│   │   ├── fill_models.py           # Order fill models
│   │   └── enhanced_reward.py       # Reward functions
│   └── utils/              # Utilities
│       ├── config.py       # Configuration loading
│       ├── metrics.py      # Performance metrics
│       └── logging.py      # Logging utilities
├── scripts/                # Executable scripts
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation scripts
│   └── analysis/           # Analysis and visualization
├── configs/                # Configuration files
│   ├── realistic_environment.yaml  # Realistic market config
│   ├── polygon_replay.yaml         # Historical replay config
│   └── ppo_improved.yaml          # PPO hyperparameters
├── data/                   # Data storage
│   ├── polygon/            # Historical market data
│   └── replay/             # Processed replay data
├── logs/                   # Training logs and results
│   └── checkpoints/        # Model checkpoints
├── artifacts/              # Analysis outputs
│   ├── traces/             # Agent behavior traces
│   └── plots/              # Visualization plots
├── notebooks/              # Jupyter notebooks
│   ├── agent_analysis.ipynb       # Agent behavior analysis
│   └── visualize_traces.ipynb     # Trace visualization
└── tests/                  # Test suite
```

## 🎯 Key Features

### Realistic Market Simulation
- **Adverse Selection**: Market moves against filled orders
- **Volatility-Aware Fills**: Fill probability decreases with volatility
- **Enhanced Inventory Costs**: Quadratic and linear inventory penalties
- **Latency Enforcement**: Realistic order execution delays
- **Slippage/Execution Price**: Price impact modeling

### Historical Data Replay
- **Polygon Data Integration**: Real market microstructure data
- **Domain Randomization**: Robust training across market conditions
- **Calibrated Fill Models**: Realistic order execution simulation
- **Multi-Day Evaluation**: Comprehensive performance testing

### RL Agent Training
- **Custom PPO Implementation**: Library-independent training
- **Observation Normalization**: Stable learning dynamics
- **Comprehensive Metrics**: PnL, Sharpe ratio, inventory management
- **Baseline Comparisons**: Avellaneda-Stoikov, Fixed Spread, Random, Inventory Mean Reversion

## 📊 Performance Results

### Synthetic Environment
| Agent | Mean PnL | Sharpe Ratio | Fill Rate | Inventory Variance |
|-------|----------|--------------|-----------|-------------------|
| **PPO RL** | **28.54** | **0.10** | **0.52** | **77.07** |
| Avellaneda-Stoikov | 0.00 | 0.00 | 0.00 | 0.00 |
| Random | -90.87 | -0.31 | 0.00 | 0.00 |
| Inventory Mean Reversion | -250.21 | -0.89 | 0.00 | 0.00 |
| Fixed Spread | -265.21 | -0.95 | 0.00 | 0.00 |

### Historical Replay Data
| Agent | Mean PnL | Sharpe Ratio | Fill Rate | Inventory Variance |
|-------|----------|--------------|-----------|-------------------|
| Avellaneda-Stoikov | **5,523.59** | 2.94 | 87.7% | 5,140.63 |
| **PPO RL** | **4,985.62** | **23.36** | **95.0%** | **3,231.52** |
| Fixed Spread | 4,784.65 | 15.04 | 91.2% | 4,641.09 |
| Random | 826.53 | 3.18 | 80.3% | 6,255.38 |
| Inventory Mean Reversion | 166.43 | 2.53 | 75.7% | 3,849.81 |

## 🔧 Configuration

### Environment Parameters
- `episode_length`: Number of steps per episode
- `max_inventory`: Maximum absolute inventory position
- `tick_size`: Minimum price increment
- `fee_bps`: Trading fees in basis points
- `latency_ticks`: Order execution delay

### PPO Hyperparameters
- `learning_rate`: Policy learning rate
- `n_steps`: Rollout buffer size
- `batch_size`: Training batch size
- `n_epochs`: Policy update epochs
- `gamma`: Discount factor
- `gae_lambda`: GAE parameter
- `clip_range`: PPO clipping parameter

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_env_core.py -v
pytest tests/test_latency.py -v
pytest tests/test_reward.py -v
pytest tests/test_min_trainer.py -v
pytest tests/test_env_rollout.py -v
```

## 📈 Analysis and Visualization

The project includes comprehensive analysis tools:

- **Agent Behavior Traces**: Per-timestep action and market state logging
- **Performance Metrics**: PnL, Sharpe ratio, inventory management, fill rates
- **Visualization Notebooks**: Interactive analysis of agent behavior
- **Baseline Comparisons**: Systematic evaluation against established strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of Gymnasium for environment interface
- Uses PyTorch for neural network implementation
- Historical data provided by Polygon.io
- Inspired by Avellaneda-Stoikov optimal market making theory