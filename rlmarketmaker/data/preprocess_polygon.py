#!/usr/bin/env python3
"""
Preprocess Polygon data for replay evaluation.

Input: Raw Polygon tick data (parquet files)
Output: Processed replay data with standardized columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def preprocess_polygon_data(
    input_path: str,
    output_path: str,
    step_ms: int = 1000,  # 1 second steps
    tick_size: float = 0.01
) -> None:
    """
    Preprocess Polygon data for replay.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        step_ms: Step size in milliseconds
        tick_size: Minimum price increment
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate additional fields
    df['best_bid'] = df['midprice'] - df['spread'] / 2
    df['best_ask'] = df['midprice'] + df['spread'] / 2
    
    # Calculate returns and rolling volatility
    df['ret'] = df['midprice'].pct_change()
    df['vol_rolling'] = df['ret'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
    
    # Calculate volume-weighted average price (VWAP) for each step
    df['vwap_step'] = df['midprice']  # Simplified - use midprice as VWAP
    
    # Calculate order book imbalance
    df['imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-6)
    
    # Calculate traded volume (from trades column)
    # Extract number of trades from the trades array
    df['num_trades'] = df['trades'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    df['traded_vol'] = df['num_trades'] * df['volume'] / (df['num_trades'] + 1e-6)
    
    # Resample to fixed time steps
    df_resampled = df.set_index('timestamp').resample(f'{step_ms}ms').agg({
        'midprice': 'last',
        'spread': 'last', 
        'best_bid': 'last',
        'best_ask': 'last',
        'bid_size': 'last',
        'ask_size': 'last',
        'num_trades': 'sum',
        'volume': 'sum',
        'volatility': 'last',
        'ret': 'last',
        'vol_rolling': 'last',
        'vwap_step': 'last',
        'imbalance': 'last',
        'traded_vol': 'sum'
    }).dropna()
    
    # Add step index
    df_resampled['step'] = range(len(df_resampled))
    
    # Calculate depth at best bid/ask
    df_resampled['depth_at_best'] = df_resampled['bid_size'] + df_resampled['ask_size']
    
    # Round prices to tick size
    df_resampled['midprice'] = np.round(df_resampled['midprice'] / tick_size) * tick_size
    df_resampled['best_bid'] = np.round(df_resampled['best_bid'] / tick_size) * tick_size
    df_resampled['best_ask'] = np.round(df_resampled['best_ask'] / tick_size) * tick_size
    df_resampled['spread'] = df_resampled['best_ask'] - df_resampled['best_bid']
    
    # Select final columns
    output_columns = [
        'step', 'midprice', 'spread', 'best_bid', 'best_ask',
        'bid_size', 'ask_size', 'traded_vol', 'vwap_step', 
        'ret', 'vol_rolling', 'imbalance', 'depth_at_best', 'num_trades'
    ]
    
    df_final = df_resampled[output_columns].reset_index()
    
    # Save to parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    
    print(f"Processed data saved to {output_path}")
    print(f"Shape: {df_final.shape}")
    print(f"Time range: {df_final['timestamp'].min()} to {df_final['timestamp'].max()}")
    print(f"Columns: {df_final.columns.tolist()}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Preprocess Polygon data for replay')
    parser.add_argument('--input', type=str, required=True, help='Input parquet file path')
    parser.add_argument('--output', type=str, required=True, help='Output parquet file path')
    parser.add_argument('--step_ms', type=int, default=1000, help='Step size in milliseconds')
    parser.add_argument('--tick_size', type=float, default=0.01, help='Minimum price increment')
    
    args = parser.parse_args()
    
    preprocess_polygon_data(
        input_path=args.input,
        output_path=args.output,
        step_ms=args.step_ms,
        tick_size=args.tick_size
    )


if __name__ == '__main__':
    main()
