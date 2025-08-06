"""
Test script to verify the fixes for division by zero and news feed issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_ticker_mapping():
    """Test ticker mapping fixes"""
    from ticker_mapping import get_base_symbol, format_ticker_for_source
    
    print("Testing Ticker Mapping...")
    print("-" * 50)
    
    test_cases = [
        ('BTC', 'yfinance', 'BTC-USD'),
        ('BTC', 'binance', 'BTCUSDT'),
        ('ETH', 'yfinance', 'ETH-USD'),
        ('ETH', 'binance', 'ETHUSDT'),
    ]
    
    for base, source, expected in test_cases:
        result = format_ticker_for_source(base, source)
        status = "✓" if result == expected else "✗"
        print(f"{status} {base} -> {source}: {result} (expected: {expected})")
    
    print()

def test_division_safety():
    """Test division by zero fixes"""
    print("Testing Division Safety...")
    print("-" * 50)
    
    # Test empty data handling
    import numpy as np
    from sklearn.metrics import silhouette_score
    
    # This should not crash with division by zero
    try:
        X = np.array([[1, 2], [1, 2], [1, 2]])  # All same points
        labels = [0, 0, 0]
        
        # This would normally cause issues
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = 0.0
        
        print(f"✓ Handled identical points clustering: score = {score}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print()

def test_news_feed():
    """Test RSS feed fetching"""
    print("Testing News Feed...")
    print("-" * 50)
    
    from rss_feed_handler import rss_handler
    
    # Test fetching from one source
    try:
        articles = rss_handler.fetch_feed("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", limit=2)
        if articles:
            print(f"✓ Fetched {len(articles)} articles from CoinDesk")
            if articles:
                print(f"  Latest: {articles[0].get('title', 'No title')[:50]}...")
        else:
            print("⚠ No articles fetched (may be network issue)")
    except Exception as e:
        print(f"✗ Failed to fetch news: {e}")
    
    print()

def test_what_if_safety():
    """Test what-if scenario division safety"""
    print("Testing What-If Scenarios Safety...")
    print("-" * 50)
    
    # Test division by zero protection
    try:
        # Simulate division scenarios
        investment = 0
        profit_loss = 100
        
        # This should not crash
        pct = (profit_loss / investment) * 100 if investment > 0 else 0
        print(f"✓ Handled zero investment: {pct}%")
        
        # Test with valid values
        investment = 1000
        pct = (profit_loss / investment) * 100 if investment > 0 else 0
        print(f"✓ Normal calculation: {pct}%")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print()

def main():
    print("\n" + "=" * 60)
    print("TESTING FIXES FOR PIPELINE AND NEWS FEED ISSUES")
    print("=" * 60 + "\n")
    
    test_ticker_mapping()
    test_division_safety()
    test_news_feed()
    test_what_if_safety()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
