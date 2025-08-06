#!/usr/bin/env python3
"""
Comprehensive test for all database-integrated endpoints
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, url, params=None, data=None, expected_status=200):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, params=params, timeout=30)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code == expected_status:
            print(f"✅ Success! Status: {response.status_code}")
            
            # Show response preview
            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
                if isinstance(data, dict):
                    # Show keys and sample values
                    for key in list(data.keys())[:3]:
                        value = data[key]
                        if isinstance(value, dict):
                            print(f"  - {key}: {len(value)} items")
                        elif isinstance(value, list):
                            print(f"  - {key}: {len(value)} items")
                        else:
                            print(f"  - {key}: {str(value)[:50]}")
            return True
        else:
            print(f"❌ Failed! Status: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
    except requests.exceptions.Timeout:
        print(f"⚠️  Timeout - endpoint took too long")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def run_tests():
    """Run all endpoint tests"""
    print("="*60)
    print("COMPREHENSIVE ENDPOINT TESTING")
    print("Testing Database Integration")
    print("="*60)
    
    results = []
    
    # Test database status
    results.append(("Database Summary", test_endpoint(
        "Database Summary",
        "GET",
        f"{BASE_URL}/database/summary"
    )))
    
    # Test data download
    results.append(("Download Single Ticker", test_endpoint(
        "Download BTC from YFinance",
        "GET",
        f"{BASE_URL}/download/BTC",
        params={"source": "yfinance", "period": "7d", "interval": "1d"}
    )))
    
    # Test signals (with database)
    results.append(("Signals API", test_endpoint(
        "Signals for BTC",
        "GET",
        f"{BASE_URL}/signals/BTC",
        params={"model_type": "arima", "periods": 7, "source": "yfinance"}
    )))
    
    # Test forecast
    results.append(("Forecast API", test_endpoint(
        "Forecast for BTC",
        "GET",
        f"{BASE_URL}/forecast/BTC",
        params={"model_type": "arima", "periods": 7, "source": "yfinance"}
    )))
    
    # Test indicators
    results.append(("Indicators API", test_endpoint(
        "Indicators for BTC",
        "GET",
        f"{BASE_URL}/indicators/BTC",
        params={"source": "yfinance"}
    )))
    
    # Test EDA (database-based)
    results.append(("EDA Analysis", test_endpoint(
        "EDA for BTC",
        "GET",
        f"{BASE_URL}/eda/BTC",
        params={"source": "yfinance"}
    )))
    
    # Test correlation (database-based)
    results.append(("Correlation Analysis", test_endpoint(
        "Correlation BTC,ETH",
        "GET",
        f"{BASE_URL}/correlation",
        params={"tickers": "BTC,ETH", "feature": "Close", "source": "yfinance"}
    )))
    
    # Test clustering (database-based)
    results.append(("Clustering Analysis", test_endpoint(
        "Clustering Analysis",
        "GET",
        f"{BASE_URL}/clustering",
        params={"source": "yfinance"}
    )))
    
    # Test backtest
    results.append(("Backtest API", test_endpoint(
        "Backtest for BTC",
        "GET",
        f"{BASE_URL}/backtest/BTC",
        params={"model_type": "arima", "periods": 7, "source": "yfinance"}
    )))
    
    # Test news feed
    results.append(("News Feed", test_endpoint(
        "Crypto News",
        "GET",
        f"{BASE_URL}/news/feed",
        params={"limit": 5}
    )))
    
    # Test what-if scenarios
    results.append(("What-If Scenario", test_endpoint(
        "Price Change Scenario",
        "GET",
        f"{BASE_URL}/whatif/price-change",
        params={"ticker": "BTC", "investment": 1000, "target_price": 50000}
    )))
    
    # Test pipeline status
    results.append(("Pipeline Status", test_endpoint(
        "Pipeline Status",
        "GET",
        f"{BASE_URL}/pipeline/status"
    )))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:.<40} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Database integration is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("\nCommon fixes:")
        print("1. Restart the server to load new code")
        print("2. Clear the database if schema changed: rm app/data/crypto_data.db")
        print("3. Check the server logs for detailed errors")
    
    return passed == total

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return True
    except:
        return False

def main():
    print("="*60)
    print("DATABASE INTEGRATION TEST SUITE")
    print("="*60)
    
    # Check if server is running
    if not check_server():
        print("\n❌ Server is not running!")
        print("\nPlease start the server first:")
        print("  cd backend")
        print("  python app/main.py")
        print("\nThen run this test again.")
        return False
    
    print("✅ Server is running on port 8000")
    print("\nStarting tests in 2 seconds...")
    time.sleep(2)
    
    return run_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
