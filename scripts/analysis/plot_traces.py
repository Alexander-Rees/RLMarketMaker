#!/usr/bin/env python3
"""Plot trace data to visualize agent behavior."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def main():
    """Main plotting function."""
    # Create output directory
    os.makedirs('artifacts/plots', exist_ok=True)
    
    # Load trace data
    ppo_trace = pd.read_csv('artifacts/traces/PPO_RL_Improved_trace.csv')
    baseline_trace = pd.read_csv('artifacts/traces/AS_trace.csv')
    
    print(f"PPO trace: {len(ppo_trace)} steps")
    print(f"Baseline trace: {len(baseline_trace)} steps")
    print(f"PPO columns: {list(ppo_trace.columns)}")
    
    # PLOT 1: Price & Quotes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot midprice
    ax.plot(ppo_trace['t'], ppo_trace['mid'], label='Midprice', linewidth=1, alpha=0.7)
    
    # Plot PPO quotes
    ax.plot(ppo_trace['t'], ppo_trace['bid_quote'], label='PPO Bid', linewidth=1, alpha=0.8)
    ax.plot(ppo_trace['t'], ppo_trace['ask_quote'], label='PPO Ask', linewidth=1, alpha=0.8)
    
    # Plot baseline quotes
    ax.plot(baseline_trace['t'], baseline_trace['bid_quote'], label='Baseline Bid', linewidth=1, alpha=0.8, linestyle='--')
    ax.plot(baseline_trace['t'], baseline_trace['ask_quote'], label='Baseline Ask', linewidth=1, alpha=0.8, linestyle='--')
    
    # Mark fills
    ppo_fills = ppo_trace[ppo_trace['filled_bid'] == 1]
    if len(ppo_fills) > 0:
        ax.scatter(ppo_fills['t'], ppo_fills['bid_quote'], color='green', marker='^', s=20, alpha=0.7, label='PPO Bid Fills')
    
    ppo_ask_fills = ppo_trace[ppo_trace['filled_ask'] == 1]
    if len(ppo_ask_fills) > 0:
        ax.scatter(ppo_ask_fills['t'], ppo_ask_fills['ask_quote'], color='red', marker='v', s=20, alpha=0.7, label='PPO Ask Fills')
    
    baseline_fills = baseline_trace[baseline_trace['filled_bid'] == 1]
    if len(baseline_fills) > 0:
        ax.scatter(baseline_fills['t'], baseline_fills['bid_quote'], color='green', marker='^', s=20, alpha=0.5, label='Baseline Bid Fills')
    
    baseline_ask_fills = baseline_trace[baseline_trace['filled_ask'] == 1]
    if len(baseline_ask_fills) > 0:
        ax.scatter(baseline_ask_fills['t'], baseline_ask_fills['ask_quote'], color='red', marker='v', s=20, alpha=0.5, label='Baseline Ask Fills')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title('Price & Quotes: PPO vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/price_quotes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # PLOT 2: Inventory
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(ppo_trace['t'], ppo_trace['inventory'], label='PPO Inventory', linewidth=2)
    ax.plot(baseline_trace['t'], baseline_trace['inventory'], label='Baseline Inventory', linewidth=2, linestyle='--')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Inventory')
    ax.set_title('Inventory Over Time: PPO vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/inventory.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # PLOT 3: Cumulative PnL
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(ppo_trace['t'], ppo_trace['cum_pnl'], label='PPO PnL', linewidth=2)
    ax.plot(baseline_trace['t'], baseline_trace['cum_pnl'], label='Baseline PnL', linewidth=2, linestyle='--')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative PnL')
    ax.set_title('Cumulative PnL: PPO vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/equity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # PLOT 4: Action Offsets Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bid offsets
    ax1.hist(ppo_trace['action_bid_off'], bins=20, alpha=0.7, label='PPO', density=True)
    ax1.hist(baseline_trace['action_bid_off'], bins=20, alpha=0.7, label='Baseline', density=True)
    ax1.set_xlabel('Bid Offset (ticks)')
    ax1.set_ylabel('Density')
    ax1.set_title('Bid Offset Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ask offsets
    ax2.hist(ppo_trace['action_ask_off'], bins=20, alpha=0.7, label='PPO', density=True)
    ax2.hist(baseline_trace['action_ask_off'], bins=20, alpha=0.7, label='Baseline', density=True)
    ax2.set_xlabel('Ask Offset (ticks)')
    ax2.set_ylabel('Density')
    ax2.set_title('Ask Offset Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/actions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary Statistics
    print("\n=== PPO Agent Performance ===")
    print(f"Final PnL: {ppo_trace['cum_pnl'].iloc[-1]:.2f}")
    print(f"Max Inventory: {ppo_trace['inventory'].max():.2f}")
    print(f"Min Inventory: {ppo_trace['inventory'].min():.2f}")
    print(f"Inventory Std: {ppo_trace['inventory'].std():.2f}")
    print(f"Total Fills: {(ppo_trace['filled_bid'] + ppo_trace['filled_ask']).sum()}")
    print(f"Fill Rate: {(ppo_trace['filled_bid'] + ppo_trace['filled_ask']).mean():.3f}")
    
    print("\n=== Baseline Agent Performance ===")
    print(f"Final PnL: {baseline_trace['cum_pnl'].iloc[-1]:.2f}")
    print(f"Max Inventory: {baseline_trace['inventory'].max():.2f}")
    print(f"Min Inventory: {baseline_trace['inventory'].min():.2f}")
    print(f"Inventory Std: {baseline_trace['inventory'].std():.2f}")
    print(f"Total Fills: {(baseline_trace['filled_bid'] + baseline_trace['filled_ask']).sum()}")
    print(f"Fill Rate: {(baseline_trace['filled_bid'] + baseline_trace['filled_ask']).mean():.3f}")
    
    print("\nPlots saved to artifacts/plots/")

if __name__ == '__main__':
    main()
