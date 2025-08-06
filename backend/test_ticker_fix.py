"""
Test script to verify ticker mapping fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from ticker_mapping import get_base_symbol, format_ticker_for_source, get_top_30_base_symbols

def test_ticker_mapping():
    print("Testing ticker mapping utilities...")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ('BTCUSDT', 'yfinance', 'BTC-USD'),
        ('BTCUSDT', 'binance', 'BTCUSDT'),
        ('BTC-USD', 'yfinance', 'BTC-USD'),
        ('BTC-USD', 'binance', 'BTCUSDT'),
        ('BTC', 'yfinance', 'BTC-USD'),
        ('BTC', 'binance', 'BTCUSDT'),
        ('ETHUSDT', 'yfinance', 'ETH-USD'),
        ('ETH', 'binance', 'ETHUSDT'),
        ('USDTUSDT', 'yfinance', 'USDT-USD'),
        ('USDT', 'binance', 'USDTBUSD'),  # Special case for USDT
    ]
    
    for input_ticker, source, expected in test_cases:
        base = get_base_symbol(input_ticker)
        formatted = format_ticker_for_source(base, source)
        status = "✓" if formatted == expected else "✗"
        print(f"{status} Input: {input_ticker:12} Source: {source:8} -> Base: {base:6} -> Output: {formatted:12} (Expected: {expected})")
    
    print("\n" + "=" * 50)
    print("Top 30 base symbols:")
    top_30 = get_top_30_base_symbols()
    print(f"Count: {len(top_30)}")
    print(f"First 10: {top_30[:10]}")
    
    print("\n" + "=" * 50)
    print("Testing download simulation...")
    
    # Simulate what happens in download_top_30_cryptos
    for base_symbol in top_30[:5]:  # Just test first 5
        yf_ticker = format_ticker_for_source(base_symbol, 'yfinance')
        bn_ticker = format_ticker_for_source(base_symbol, 'binance')
        print(f"Base: {base_symbol:6} -> YFinance: {yf_ticker:12} Binance: {bn_ticker:12}")

if __name__ == "__main__":
    test_ticker_mapping()
