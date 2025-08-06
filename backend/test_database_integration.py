#!/usr/bin/env python3
"""
Test script to verify database integration for all analysis modules
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import crypto_db, get_db_connection
from app.unified_data_handler import unified_handler
from app.signals import generate_signals, check_existing_signals, store_signals_to_db
from app.correlation_analysis import perform_correlation_analysis
from app.eda_analysis import perform_eda_analysis
from app.grouping_analysis import perform_dimensionality_reduction, perform_clustering_analysis
import pandas as pd


def test_database_setup():
    """Test database initialization"""
    print("\n1. Testing Database Setup...")
    try:
        crypto_db.initialize_database()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"   ✓ Database initialized with {len(tables)} tables")
            for table in tables:
                print(f"     - {table[0]}")
        return True
    except Exception as e:
        print(f"   ✗ Database setup failed: {e}")
        return False


async def test_data_download():
    """Test downloading data to database"""
    print("\n2. Testing Data Download to Database...")
    try:
        # Download sample data
        ticker = "BTC"
        await unified_handler.download_and_store_data(
            ticker=ticker,
            source="yfinance",
            period="30d",
            interval="1d"
        )
        
        # Verify data in database
        df = unified_handler.get_data_from_db(ticker, source="yfinance")
        if df is not None and not df.empty:
            print(f"   ✓ Downloaded {len(df)} records for {ticker}")
            print(f"     Columns: {list(df.columns)}")
            return True
        else:
            print(f"   ✗ No data found for {ticker}")
            return False
    except Exception as e:
        print(f"   ✗ Data download failed: {e}")
        return False


def test_signals_database():
    """Test signals storage and retrieval"""
    print("\n3. Testing Signals Database Integration...")
    try:
        # Get sample data
        ticker = "BTC"
        df = unified_handler.get_data_from_db(ticker, source="yfinance")
        
        if df is None or df.empty:
            print("   ⚠ No data available for signals test")
            return False
        
        # Generate sample forecast
        prices = df["Close"] if "Close" in df.columns else df["close"]
        forecast = prices.iloc[-7:] * 1.05  # Simple 5% increase forecast
        
        # Generate and store signals
        signals_df = generate_signals(forecast, threshold=0.01)
        store_signals_to_db(ticker, signals_df, "test_model", prices)
        
        # Check retrieval
        existing = check_existing_signals(ticker, "test_model")
        if existing:
            print(f"   ✓ Signals stored and retrieved: {len(existing)} records")
            return True
        else:
            print("   ✗ Failed to retrieve signals")
            return False
    except Exception as e:
        print(f"   ✗ Signals test failed: {e}")
        return False


def test_correlation_database():
    """Test correlation analysis with database"""
    print("\n4. Testing Correlation Analysis Database Integration...")
    try:
        # Download data for multiple tickers
        tickers = ["BTC", "ETH"]
        for ticker in tickers:
            df = unified_handler.get_data_from_db(ticker, source="yfinance")
            if df is None or df.empty:
                print(f"   ⚠ Downloading data for {ticker}...")
                asyncio.run(unified_handler.download_and_store_data(
                    ticker=ticker,
                    source="yfinance",
                    period="30d",
                    interval="1d"
                ))
        
        # Perform correlation analysis
        corr_df, report, fig = perform_correlation_analysis(
            use_database=True,
            selected_tickers=tickers,
            feature="Close"
        )
        
        if corr_df is not None and not corr_df.empty:
            print(f"   ✓ Correlation analysis completed")
            print(f"     Matrix shape: {corr_df.shape}")
            return True
        else:
            print("   ✗ Correlation analysis failed")
            return False
    except Exception as e:
        print(f"   ✗ Correlation test failed: {e}")
        return False


def test_eda_database():
    """Test EDA with database"""
    print("\n5. Testing EDA Database Integration...")
    try:
        ticker = "BTC"
        
        # Perform EDA using database
        report, charts = perform_eda_analysis(
            ticker=ticker,
            use_database=True
        )
        
        if report and "num_records" in report:
            print(f"   ✓ EDA completed for {ticker}")
            print(f"     Records analyzed: {report['num_records']}")
            print(f"     Charts generated: {len(charts)}")
            return True
        else:
            print("   ✗ EDA failed")
            return False
    except Exception as e:
        print(f"   ✗ EDA test failed: {e}")
        return False


def test_clustering_database():
    """Test clustering with database"""
    print("\n6. Testing Clustering Database Integration...")
    try:
        # First perform dimensionality reduction
        reduced_df, report, best_algo, fig = perform_dimensionality_reduction(
            use_database=True,
            source="yfinance"
        )
        
        if reduced_df is not None and not reduced_df.empty:
            print(f"   ✓ Dimensionality reduction completed")
            print(f"     Algorithm: {best_algo}")
            
            # Then perform clustering
            cluster_df, cluster_report, cluster_fig = perform_clustering_analysis(
                use_database=True,
                source="yfinance"
            )
            
            if cluster_df is not None:
                print(f"   ✓ Clustering completed")
                print(f"     Method: {cluster_report.get('chosen_method', 'Unknown')}")
                return True
        
        print("   ✗ Clustering failed")
        return False
    except Exception as e:
        print(f"   ✗ Clustering test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DATABASE INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Database Setup", test_database_setup()))
    results.append(("Data Download", asyncio.run(test_data_download())))
    results.append(("Signals Storage", test_signals_database()))
    results.append(("Correlation Analysis", test_correlation_database()))
    results.append(("EDA Analysis", test_eda_database()))
    results.append(("Clustering Analysis", test_clustering_database()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Database integration is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
