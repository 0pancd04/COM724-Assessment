#!/usr/bin/env python3
"""
Test the fixed signals API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_signals_api():
    """Test the signals API with various tickers"""
    
    test_cases = [
        {"ticker": "BTC", "model_type": "arima", "periods": 7},
        {"ticker": "ETH", "model_type": "sarima", "periods": 5},
        {"ticker": "BCH", "model_type": "rf", "periods": 3},  # This was failing before
    ]
    
    for test in test_cases:
        ticker = test["ticker"]
        model_type = test["model_type"]
        periods = test["periods"]
        
        print(f"\n{'='*60}")
        print(f"Testing: {ticker} with {model_type} for {periods} periods")
        print('='*60)
        
        try:
            # Test signals endpoint
            url = f"{BASE_URL}/signals/{ticker}"
            params = {
                "model_type": model_type,
                "periods": periods,
                "threshold": 0.01,
                "source": "yfinance"
            }
            
            print(f"Calling: {url}")
            print(f"Params: {params}")
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Success!")
                print(f"  - From cache: {data.get('from_cache', False)}")
                print(f"  - Signals count: {len(data.get('signals', {}))}")
                print(f"  - PnL entries: {len(data.get('pnl', {}))}")
                
                # Show first few signals
                signals = data.get('signals', {})
                if signals:
                    print(f"  - First signals:")
                    for date, signal in list(signals.items())[:3]:
                        print(f"    {date}: {signal}")
            else:
                print(f"✗ Failed with status {response.status_code}")
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    # Test other related endpoints
    print(f"\n{'='*60}")
    print("Testing other endpoints")
    print('='*60)
    
    # Test forecast endpoint
    try:
        url = f"{BASE_URL}/forecast/BTC"
        params = {"model_type": "arima", "periods": 7, "source": "yfinance"}
        print(f"\n1. Testing forecast: {url}")
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("   ✓ Forecast endpoint working")
        else:
            print(f"   ✗ Forecast failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Forecast exception: {e}")
    
    # Test indicators endpoint
    try:
        url = f"{BASE_URL}/indicators/BTC"
        params = {"source": "yfinance"}
        print(f"\n2. Testing indicators: {url}")
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Indicators endpoint working")
            print(f"     - RSI entries: {len(data.get('rsi', {}))}")
            print(f"     - MACD entries: {len(data.get('macd', {}))}")
        else:
            print(f"   ✗ Indicators failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Indicators exception: {e}")
    
    # Test backtest endpoint
    try:
        url = f"{BASE_URL}/backtest/BTC"
        params = {"model_type": "arima", "periods": 7, "threshold": 0.01, "source": "yfinance"}
        print(f"\n3. Testing backtest: {url}")
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("   ✓ Backtest endpoint working")
        else:
            print(f"   ✗ Backtest failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Backtest exception: {e}")

if __name__ == "__main__":
    print("Testing Fixed Signals API")
    print("="*60)
    print("Make sure the backend server is running on port 8000")
    print("="*60)
    
    test_signals_api()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
