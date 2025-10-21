"""Logging utilities for market making training and evaluation."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class EpisodeLogger:
    """Log episode-level metrics to CSV."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize episode logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"run_{timestamp}.csv"
        
        # Initialize CSV writer
        self.fieldnames = [
            'episode', 'pnl', 'sharpe', 'inv_var', 'fill_rate', 'max_dd',
            'episode_length', 'mean_inventory', 'std_inventory', 'total_trades'
        ]
        
        self.csv_file = open(self.log_file, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        self.episode_count = 0
    
    def log_episode(self, 
                   pnl: float,
                   sharpe: float,
                   inv_var: float,
                   fill_rate: float,
                   max_dd: float,
                   episode_length: int,
                   mean_inventory: float,
                   std_inventory: float,
                   total_trades: int):
        """
        Log episode metrics.
        
        Args:
            pnl: Episode PnL
            sharpe: Sharpe ratio for episode
            inv_var: Inventory variance
            fill_rate: Fill rate (0-1)
            max_dd: Maximum drawdown
            episode_length: Number of steps in episode
            mean_inventory: Mean inventory during episode
            std_inventory: Standard deviation of inventory
            total_trades: Total number of trades
        """
        self.episode_count += 1
        
        row = {
            'episode': self.episode_count,
            'pnl': pnl,
            'sharpe': sharpe,
            'inv_var': inv_var,
            'fill_rate': fill_rate,
            'max_dd': max_dd,
            'episode_length': episode_length,
            'mean_inventory': mean_inventory,
            'std_inventory': std_inventory,
            'total_trades': total_trades
        }
        
        self.writer.writerow(row)
        self.csv_file.flush()
    
    def close(self):
        """Close the log file."""
        self.csv_file.close()
    
    def get_log_path(self) -> str:
        """Get path to log file."""
        return str(self.log_file)


class EvaluationLogger:
    """Log evaluation results to JSON."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize evaluation logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def log_evaluation(self, 
                      agent_name: str,
                      metrics: Dict[str, float],
                      config: Dict[str, Any],
                      timestamp: Optional[str] = None) -> str:
        """
        Log evaluation results.
        
        Args:
            agent_name: Name of the agent being evaluated
            metrics: Dictionary of metrics
            config: Configuration used
            timestamp: Optional timestamp string
            
        Returns:
            Path to the log file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_file = self.log_dir / f"evaluation_{agent_name}_{timestamp}.json"
        
        log_data = {
            'agent_name': agent_name,
            'timestamp': timestamp,
            'metrics': metrics,
            'config': config
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return str(log_file)
    
    def log_comparison(self, 
                      results: Dict[str, Dict[str, float]],
                      config: Dict[str, Any],
                      timestamp: Optional[str] = None) -> str:
        """
        Log comparison results between multiple agents.
        
        Args:
            results: Dictionary mapping agent names to their metrics
            config: Configuration used
            timestamp: Optional timestamp string
            
        Returns:
            Path to the log file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_file = self.log_dir / f"comparison_{timestamp}.json"
        
        log_data = {
            'timestamp': timestamp,
            'results': results,
            'config': config
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return str(log_file)


class TrainingLogger:
    """Log training progress and metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.csv"
        
        # Initialize CSV writer
        self.fieldnames = [
            'step', 'episode', 'reward', 'loss', 'learning_rate', 'entropy',
            'value_loss', 'policy_loss', 'explained_variance'
        ]
        
        self.csv_file = open(self.log_file, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        self.step_count = 0
        self.episode_count = 0
    
    def log_training_step(self, 
                         reward: float,
                         loss: float,
                         learning_rate: float,
                         entropy: float,
                         value_loss: float,
                         policy_loss: float,
                         explained_variance: float):
        """
        Log training step metrics.
        
        Args:
            reward: Reward for this step
            loss: Total loss
            learning_rate: Current learning rate
            entropy: Policy entropy
            value_loss: Value function loss
            policy_loss: Policy loss
            explained_variance: Explained variance
        """
        self.step_count += 1
        
        row = {
            'step': self.step_count,
            'episode': self.episode_count,
            'reward': reward,
            'loss': loss,
            'learning_rate': learning_rate,
            'entropy': entropy,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'explained_variance': explained_variance
        }
        
        self.writer.writerow(row)
        self.csv_file.flush()
    
    def log_episode(self, episode: int):
        """Log episode completion."""
        self.episode_count = episode
    
    def close(self):
        """Close the log file."""
        self.csv_file.close()
    
    def get_log_path(self) -> str:
        """Get path to log file."""
        return str(self.log_file)


def load_episode_logs(log_file: str) -> pd.DataFrame:
    """
    Load episode logs from CSV file.
    
    Args:
        log_file: Path to CSV log file
        
    Returns:
        DataFrame with episode data
    """
    return pd.read_csv(log_file)


def load_evaluation_logs(log_file: str) -> Dict[str, Any]:
    """
    Load evaluation logs from JSON file.
    
    Args:
        log_file: Path to JSON log file
        
    Returns:
        Dictionary with evaluation data
    """
    with open(log_file, 'r') as f:
        return json.load(f)


def create_summary_report(log_dir: str, output_file: str = "summary_report.json"):
    """
    Create summary report from all logs in directory.
    
    Args:
        log_dir: Directory containing log files
        output_file: Output file name for summary report
    """
    log_dir = Path(log_dir)
    
    # Find all CSV log files
    csv_files = list(log_dir.glob("run_*.csv"))
    
    if not csv_files:
        print("No CSV log files found")
        return
    
    # Load and combine all episode logs
    all_episodes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_episodes.append(df)
    
    combined_df = pd.concat(all_episodes, ignore_index=True)
    
    # Calculate summary statistics
    summary = {
        'total_episodes': len(combined_df),
        'total_steps': combined_df['episode_length'].sum(),
        'mean_pnl': combined_df['pnl'].mean(),
        'std_pnl': combined_df['pnl'].std(),
        'mean_sharpe': combined_df['sharpe'].mean(),
        'mean_fill_rate': combined_df['fill_rate'].mean(),
        'mean_max_drawdown': combined_df['max_dd'].mean(),
        'best_episode_pnl': combined_df['pnl'].max(),
        'worst_episode_pnl': combined_df['pnl'].min()
    }
    
    # Save summary report
    output_path = log_dir / output_file
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary report saved to {output_path}")
    return str(output_path)
