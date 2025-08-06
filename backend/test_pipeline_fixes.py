"""
Test script to verify pipeline fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import asyncio
from unified_data_handler import unified_handler
from ticker_mapping import format_ticker_for_source, get_base_symbol

def test_ticker_download():
    """Test downloading data with proper ticker format"""
    print("Testing Ticker Download...")
    print("-" * 50)
    
    # Test downloading a single ticker
    test_cases = [
        ("BTC", "yfinance", "7d"),
        ("ETH", "binance", "7d"),
    ]
    
    for base_symbol, source, period in test_cases:
        try:
            print(f"\nDownloading {base_symbol} from {source}...")
            
            # This should handle ticker formatting internally
            df = unified_handler.download_and_store_data(
                base_symbol,  # Pass base symbol
                source, 
                period=period,
                interval="1d"
            )
            
            if not df.empty:
                print(f"  ✓ Downloaded {len(df)} records")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
            else:
                print(f"  ⚠ No data downloaded")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print()

def test_data_preparation():
    """Test data preparation for preprocessing"""
    print("Testing Data Preparation...")
    print("-" * 50)
    
    sources = ["yfinance", "binance"]
    
    for source in sources:
        print(f"\nPreparing data for {source}...")
        try:
            df = unified_handler.prepare_data_for_preprocessing(source)
            
            if not df.empty:
                print(f"  ✓ Prepared data shape: {df.shape}")
                print(f"  Tickers: {list(df.index[:3])}...")
            else:
                print(f"  ⚠ No data prepared (might need to download first)")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print()

def test_pipeline_step():
    """Test a pipeline step"""
    print("Testing Pipeline Step...")
    print("-" * 50)
    
    from pipeline_orchestrator import DataPreprocessingStep
    
    async def run_test():
        # Test preprocessing for both sources
        for source in ["yfinance", "binance"]:
            print(f"\nTesting preprocessing for {source}...")
            
            step = DataPreprocessingStep(source, max_days=7)
            try:
                result = await step.execute()
                
                if result.get('skipped'):
                    print(f"  ⚠ Step skipped: {result.get('reason')}")
                else:
                    print(f"  ✓ Preprocessing completed")
                    print(f"  Output: {result.get('output_file')}")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Run async test
    asyncio.run(run_test())
    print()

def main():
    print("\n" + "=" * 60)
    print("TESTING PIPELINE FIXES")
    print("=" * 60 + "\n")
    
    test_ticker_download()
    test_data_preparation()
    test_pipeline_step()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Clear the database if needed: rm app/crypto_data.db")
    print("2. Run the full pipeline from the dashboard")
    print("3. Use shorter periods (7d-30d) for faster testing")

if __name__ == "__main__":
    main()
