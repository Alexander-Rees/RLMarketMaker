"""Binance data preprocessing for market making."""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
import requests
import time
from datetime import datetime, timedelta


class BinanceDataProcessor:
    """Processes Binance historical data for market making."""
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m"):
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://api.binance.com/api/v3/klines"
        
    def fetch_data(self, 
                   start_date: str, 
                   end_date: str, 
                   output_path: str = None) -> pd.DataFrame:
        """
        Fetch historical kline data from Binance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_path: Path to save the data (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {self.symbol} data from {start_date} to {end_date}...")
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                # Binance API limit: 1000 klines per request
                params = {
                    'symbol': self.symbol,
                    'interval': self.interval,
                    'startTime': current_ts,
                    'endTime': min(current_ts + 1000 * 60 * 1000, end_ts),  # 1000 minutes max
                    'limit': 1000
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                if not data:
                    break
                    
                all_data.extend(data)
                current_ts = data[-1][0] + 1  # Next timestamp
                
                print(f"Fetched {len(data)} candles, total: {len(all_data)}")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_data:
            raise ValueError("No data fetched")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].astype(float)
        
        # Calculate midprice and spread
        df['midprice'] = (df['high'] + df['low']) / 2
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['midprice']
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate volatility (rolling 20-period)
        df['volatility'] = df['log_returns'].rolling(20).std() * np.sqrt(1440)  # Annualized
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"Processed {len(df)} data points")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        if output_path:
            df.to_parquet(output_path, index=False)
            print(f"Data saved to {output_path}")
        
        return df
    
    def create_market_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic market ticks from OHLCV data.
        This simulates order book data by adding noise to midprice.
        """
        ticks = []
        
        for _, row in df.iterrows():
            # Create multiple ticks per minute to simulate order book updates
            n_ticks = max(1, int(row['volume'] / 100))  # More ticks for higher volume
            
            for i in range(n_ticks):
                # Add noise to midprice to simulate bid/ask spread
                noise = np.random.normal(0, row['spread'] * 0.1)
                tick_price = row['midprice'] + noise
                
                # Simulate bid/ask sizes
                bid_size = np.random.uniform(10, 100)
                ask_size = np.random.uniform(10, 100)
                
                # Simulate trades (Poisson process)
                n_trades = np.random.poisson(row['trades'] / n_ticks)
                trades = []
                for _ in range(n_trades):
                    trade_size = np.random.uniform(1, 10)
                    trade_price = tick_price + np.random.uniform(-row['spread']/2, row['spread']/2)
                    trades.append({'size': trade_size, 'price': trade_price})
                
                ticks.append({
                    'timestamp': row['timestamp'] + pd.Timedelta(seconds=i*60/n_ticks),
                    'midprice': tick_price,
                    'spread': row['spread'],
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'trades': trades,
                    'volume': row['volume'],
                    'volatility': row['volatility']
                })
        
        return pd.DataFrame(ticks)


def download_sample_data():
    """Download a small sample of BTC data for testing."""
    processor = BinanceDataProcessor("BTCUSDT", "1m")
    
    # Download last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Fetch and save data
    df = processor.fetch_data(start_str, end_str)
    output_path = 'data/btcusdt_sample.parquet'
    df.to_parquet(output_path, index=False)
    
    # Create market ticks
    ticks_df = processor.create_market_ticks(df)
    ticks_path = 'data/btcusdt_ticks.parquet'
    ticks_df.to_parquet(ticks_path, index=False)
    
    print(f"Sample data saved to {output_path}")
    print(f"Market ticks saved to {ticks_path}")
    
    return df, ticks_df


if __name__ == "__main__":
    # Download sample data
    df, ticks_df = download_sample_data()
    
    print("\nData summary:")
    print(f"OHLCV data: {len(df)} rows")
    print(f"Market ticks: {len(ticks_df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
