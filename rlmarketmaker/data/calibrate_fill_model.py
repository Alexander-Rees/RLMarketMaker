#!/usr/bin/env python3
"""
Calibrate fill model parameters from historical data.

This script analyzes historical market data to fit fill model parameters
that match observed fill rates and hit rates.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Tuple
import argparse
import yaml
from pathlib import Path


def calculate_empirical_fill_rates(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate empirical fill rates from historical data."""
    
    # Calculate distance from best bid/ask for each trade
    data['distance_bid'] = np.maximum(0, (data['best_bid'] - data['midprice']) / 0.01)
    data['distance_ask'] = np.maximum(0, (data['midprice'] - data['best_ask']) / 0.01)
    
    # Calculate hit rates at best bid/ask
    hit_rate_bid = (data['distance_bid'] == 0).mean()
    hit_rate_ask = (data['distance_ask'] == 0).mean()
    
    # Calculate fill rates by distance
    distance_bins = np.arange(0, 10, 0.5)
    fill_rates_bid = []
    fill_rates_ask = []
    
    for i in range(len(distance_bins) - 1):
        mask_bid = (data['distance_bid'] >= distance_bins[i]) & (data['distance_bid'] < distance_bins[i+1])
        mask_ask = (data['distance_ask'] >= distance_bins[i]) & (data['distance_ask'] < distance_bins[i+1])
        
        if mask_bid.sum() > 0:
            fill_rates_bid.append(mask_bid.mean())
        else:
            fill_rates_bid.append(0.0)
            
        if mask_ask.sum() > 0:
            fill_rates_ask.append(mask_ask.mean())
        else:
            fill_rates_ask.append(0.0)
    
    return {
        'hit_rate_bid': hit_rate_bid,
        'hit_rate_ask': hit_rate_ask,
        'distance_bins': distance_bins[:-1],
        'fill_rates_bid': fill_rates_bid,
        'fill_rates_ask': fill_rates_ask
    }


def exponential_fill_model(distance: float, alpha: float, beta: float) -> float:
    """Exponential fill model: p = alpha * exp(-beta * distance)."""
    return alpha * np.exp(-beta * distance)


def objective_function(params: np.ndarray, empirical_data: Dict[str, Any]) -> float:
    """Objective function for fill model calibration."""
    alpha, beta = params
    
    # Calculate model predictions
    distances = empirical_data['distance_bins']
    model_fill_rates = [exponential_fill_model(d, alpha, beta) for d in distances]
    
    # Calculate MSE between model and empirical data
    mse_bid = np.mean([(m - e)**2 for m, e in zip(model_fill_rates, empirical_data['fill_rates_bid'])])
    mse_ask = np.mean([(m - e)**2 for m, e in zip(model_fill_rates, empirical_data['fill_rates_ask'])])
    
    # Penalty for hit rate mismatch
    hit_rate_penalty = 0.0
    if alpha > 0:
        model_hit_rate = exponential_fill_model(0, alpha, beta)
        hit_rate_penalty = (model_hit_rate - empirical_data['hit_rate_bid'])**2
    
    return mse_bid + mse_ask + hit_rate_penalty


def calibrate_fill_model(data_path: str, symbol: str) -> Dict[str, float]:
    """Calibrate fill model parameters for a given symbol."""
    
    print(f"Calibrating fill model for {symbol}...")
    print(f"Loading data from {data_path}...")
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Calculate empirical fill rates
    empirical_data = calculate_empirical_fill_rates(df)
    
    print(f"Empirical hit rate (bid): {empirical_data['hit_rate_bid']:.3f}")
    print(f"Empirical hit rate (ask): {empirical_data['hit_rate_ask']:.3f}")
    
    # Initial parameter guess
    initial_params = np.array([0.8, 0.5])  # alpha, beta
    
    # Bounds for parameters
    bounds = [(0.1, 1.0), (0.1, 2.0)]  # alpha: [0.1, 1.0], beta: [0.1, 2.0]
    
    # Optimize parameters
    result = minimize(
        objective_function,
        initial_params,
        args=(empirical_data,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    if result.success:
        alpha, beta = result.x
        print(f"Optimization successful!")
        print(f"Alpha: {alpha:.3f}")
        print(f"Beta: {beta:.3f}")
        print(f"Final objective: {result.fun:.6f}")
        
        # Calculate final hit rate
        final_hit_rate = exponential_fill_model(0, alpha, beta)
        print(f"Model hit rate: {final_hit_rate:.3f}")
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'hit_rate_bid': float(empirical_data['hit_rate_bid']),
            'hit_rate_ask': float(empirical_data['hit_rate_ask']),
            'model_hit_rate': float(final_hit_rate)
        }
    else:
        print(f"Optimization failed: {result.message}")
        return {
            'alpha': 0.8,
            'beta': 0.5,
            'hit_rate_bid': float(empirical_data['hit_rate_bid']),
            'hit_rate_ask': float(empirical_data['hit_rate_ask']),
            'model_hit_rate': 0.8
        }


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Calibrate fill model parameters')
    parser.add_argument('--data_path', type=str, required=True, help='Path to historical data')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol name (e.g., AAPL)')
    parser.add_argument('--output', type=str, default='configs/fill_calibration.yaml', 
                       help='Output configuration file')
    
    args = parser.parse_args()
    
    # Calibrate parameters
    params = calibrate_fill_model(args.data_path, args.symbol)
    
    # Save to configuration file
    config = {
        'fill_model': {
            'alpha': params['alpha'],
            'beta': params['beta'],
            'calibration': {
                'symbol': args.symbol,
                'hit_rate_bid': params['hit_rate_bid'],
                'hit_rate_ask': params['hit_rate_ask'],
                'model_hit_rate': params['model_hit_rate']
            }
        }
    }
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {args.output}")


if __name__ == '__main__':
    main()
