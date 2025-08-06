"""
Quick verification script for pipeline execution
"""

import requests
import json
import time

def test_pipeline():
    """Test the pipeline endpoint with minimal data"""
    
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("PIPELINE VERIFICATION TEST")
    print("=" * 60)
    
    # Test 1: Check API health
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✓ API is running")
        else:
            print(f"   ✗ API returned status {response.status_code}")
    except Exception as e:
        print(f"   ✗ Cannot connect to API: {e}")
        print("   Make sure the backend is running: python app/main.py")
        return
    
    # Test 2: Check database status
    print("\n2. Checking database...")
    try:
        response = requests.get(f"{base_url}/database/summary")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Database connected")
            print(f"   - Total records: {data.get('total_records', 0)}")
            print(f"   - Unique tickers: {data.get('unique_tickers', 0)}")
        else:
            print(f"   ⚠ Database endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ✗ Database check failed: {e}")
    
    # Test 3: Test minimal pipeline
    print("\n3. Testing minimal pipeline execution...")
    print("   (This will download data for 3 coins over 7 days)")
    
    params = {
        "tickers": "BTC,ETH,ADA",  # Just 3 coins
        "sources": "yfinance",       # Single source for speed
        "period": "7d",              # Short period
        "interval": "1d",
        "include_eda": "false",      # Skip heavy processing
        "include_clustering": "false"
    }
    
    try:
        print("   Starting pipeline...")
        response = requests.post(
            f"{base_url}/pipeline/full",
            params=params,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Pipeline executed successfully!")
            
            # Show results
            if "pipeline_results" in result:
                results = result["pipeline_results"]
                if "results" in results:
                    steps = results["results"]
                    print(f"   - Completed steps: {len(steps)}")
                    for step_name in steps:
                        print(f"     • {step_name}")
            
        elif response.status_code == 409:
            print("   ⚠ Another pipeline is already running")
            print("   Wait for it to complete or restart the backend")
        else:
            print(f"   ✗ Pipeline failed with status {response.status_code}")
            error = response.json().get("detail", "Unknown error")
            print(f"   Error: {error}")
            
    except requests.Timeout:
        print("   ⚠ Pipeline is taking longer than expected")
        print("   Check the backend logs for progress")
    except Exception as e:
        print(f"   ✗ Pipeline test failed: {e}")
    
    # Test 4: Test news feed
    print("\n4. Testing news feed...")
    try:
        response = requests.get(
            f"{base_url}/news/feed",
            params={"limit": 5, "refresh": "false"}
        )
        
        if response.status_code == 200:
            data = response.json()
            count = data.get("count", 0)
            if count > 0:
                print(f"   ✓ News feed working - {count} articles fetched")
            else:
                print("   ⚠ News feed returned but no articles found")
                print("   This might be due to network restrictions")
        else:
            print(f"   ✗ News feed failed with status {response.status_code}")
    except Exception as e:
        print(f"   ✗ News feed test failed: {e}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. If pipeline failed, check backend logs for details")
    print("2. Try the full pipeline from the dashboard UI")
    print("3. Use smaller date ranges if memory is limited")

if __name__ == "__main__":
    test_pipeline()
