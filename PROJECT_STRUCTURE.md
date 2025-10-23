# RLMarketMaker - Clean Project Structure

## 📁 Directory Organization

### Core Package (`rlmarketmaker/`)
- **`agents/`** - RL agents and baseline strategies
  - `min_ppo.py` - Custom PPO implementation
  - `baselines.py` - Avellaneda-Stoikov, Fixed Spread, Random, Inventory Mean Reversion
- **`data/`** - Data feeds and preprocessing
  - `feeds.py` - Synthetic, Polygon, and Binance data feeds
  - `preprocess_polygon.py` - Historical data preprocessing
  - `calibrate_fill_model.py` - Fill model parameter calibration
- **`env/`** - Market environments
  - `realistic_market_env.py` - Realistic simulation with adverse selection
  - `replay_market_env.py` - Historical data replay environment
  - `fill_models.py` - Order execution models
  - `enhanced_reward.py` - Advanced reward functions
- **`utils/`** - Utilities and helpers
  - `config.py` - Configuration management
  - `metrics.py` - Performance metrics calculation
  - `logging.py` - Logging utilities

### Scripts (`scripts/`)
- **`training/`** - Training scripts
  - `train_min.py` - Main PPO training script
- **`evaluation/`** - Evaluation scripts
  - `eval_min.py` - Synthetic environment evaluation
  - `evaluate_realistic.py` - Realistic environment evaluation
  - `evaluate_replay.py` - Historical replay evaluation
- **`analysis/`** - Analysis and visualization
  - `trace_eval.py` - Agent behavior tracing
  - `trace_replay.py` - Replay environment tracing
  - `plot_traces.py` - Trace visualization

### Configuration (`configs/`)
- `realistic_environment.yaml` - Realistic market simulation config
- `polygon_replay.yaml` - Historical replay configuration
- `ppo_improved.yaml` - Optimized PPO hyperparameters
- `api_keys.yaml` - API keys (if needed)

### Data Storage (`data/`)
- `polygon/` - Raw historical market data (AAPL, MSFT)
- `replay/` - Processed replay data

### Outputs (`logs/`, `artifacts/`)
- `logs/checkpoints/` - Trained model checkpoints
- `logs/*.json` - Evaluation results
- `artifacts/traces/` - Agent behavior traces
- `artifacts/plots/` - Visualization outputs

### Analysis (`notebooks/`)
- `agent_analysis.ipynb` - Comprehensive agent behavior analysis
- `visualize_traces.ipynb` - Trace visualization notebook

### Testing (`tests/`)
- `test_env_core.py` - Environment core functionality
- `test_env_rollout.py` - Environment rollout tests
- `test_latency.py` - Latency enforcement tests
- `test_reward.py` - Reward function tests
- `test_min_trainer.py` - PPO trainer tests

## 🧹 Cleanup Summary

### Removed Files
- **Deprecated configs**: `ppo_simple.yaml`, `ppo_tuned.yaml`, `tuned_environment.yaml`, `balanced.yaml`, `real_aapl.yaml`, `binance.yaml`, `synthetic.yaml`
- **Old scripts**: `evaluate.py`, `train.py` (replaced by organized scripts)
- **Temporary files**: Various CSV logs, tensorboard logs, eval logs
- **Redundant tests**: `test_train_smoke.py`, `test_env_api.py`

### Organized Files
- **Moved evaluation scripts** to `scripts/evaluation/`
- **Moved training scripts** to `scripts/training/`
- **Moved analysis scripts** to `scripts/analysis/`
- **Kept essential configs** only

### Key Benefits
1. **Clear separation** of training, evaluation, and analysis
2. **Reduced clutter** with only essential files
3. **Logical organization** by functionality
4. **Easy navigation** with descriptive directory names
5. **Maintainable structure** for future development

## 🚀 Usage

```bash
# Training
make train

# Evaluation
make eval
make eval-replay

# Analysis
python scripts/analysis/trace_eval.py --agent ppo --ckpt logs/checkpoints/policy.pt
python scripts/analysis/plot_traces.py

# Testing
make test
```

The project is now clean, well-organized, and ready for production use! 🎉
