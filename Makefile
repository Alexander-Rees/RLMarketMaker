# RL Market Maker Makefile
# Hermetic training pipeline

.PHONY: venv train eval test clean help

# Default target
help:
	@echo "RL Market Maker - Available targets:"
	@echo "  make venv     - Create and activate virtual environment"
	@echo "  make train    - Run training with default config"
	@echo "  make eval     - Run evaluation on latest checkpoint"
	@echo "  make test     - Run smoke tests"
	@echo "  make clean    - Clean up logs and checkpoints"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Activating virtual environment..."
	@echo "Run: source venv/bin/activate"
	@echo "Then: pip install -r requirements.txt"

# Install dependencies (run after activating venv)
install:
	pip install -r requirements.txt

# Run training
train:
	@echo "Starting RL training..."
	python scripts/training/train_min.py --config configs/realistic_environment.yaml --seed 42

# Run evaluation
eval:
	@echo "Running evaluation..."
	python scripts/evaluation/eval_min.py --checkpoint logs/checkpoints/policy.pt --config configs/realistic_environment.yaml --seed 43

# Run replay evaluation
eval-replay:
	@echo "Running replay evaluation..."
	python scripts/evaluation/evaluate_replay.py --config configs/polygon_replay.yaml --checkpoint logs/checkpoints/policy.pt --episodes 10

# Run tests
test:
	@echo "Running smoke tests..."
	python -m pytest tests/test_min_trainer.py tests/test_env_rollout.py -v

# Clean up
clean:
	@echo "Cleaning up logs and checkpoints..."
	rm -rf logs/checkpoints/*.pt
	rm -rf logs/checkpoints/*.pkl
	rm -rf logs/*.csv
	rm -rf logs/tensorboard/*

# Full setup (create venv, install, train)
setup: venv
	@echo "Setup complete. Run:"
	@echo "  source venv/bin/activate"
	@echo "  make install"
	@echo "  make train"
