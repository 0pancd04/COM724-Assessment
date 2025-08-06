#!/usr/bin/env python
"""
Startup script for the Cryptocurrency Prediction Platform
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    print("Checking requirements...")
    try:
        import fastapi
        import pandas
        import sklearn
        import plotly
        import yfinance
        print("✅ Core requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("\nPlease install requirements:")
        print("  cd backend && pip install -r requirements.txt")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path("backend/.env")
    if not env_path.exists():
        print("\n📝 Creating .env file...")
        env_content = """# Binance API Configuration (Optional)
# Get your API keys from https://www.binance.com/en/my/settings/api-management
# Uncomment and fill in your keys if you want to use Binance data

# BINANCE_API_KEY=your_api_key_here
# BINANCE_API_SECRET=your_api_secret_here
"""
        env_path.write_text(env_content)
        print("✅ .env file created (Binance API keys are optional)")

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Cryptocurrency Prediction Platform...")
    print("=" * 60)
    
    # Change to backend directory
    os.chdir("backend")
    
    # Start uvicorn
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    print("\n✅ Server starting...")
    print("\n📍 Access points:")
    print("   - API: http://localhost:8000")
    print("   - Documentation: http://localhost:8000/docs")
    print("   - Alternative docs: http://localhost:8000/redoc")
    print("\n📊 Key endpoints to get started:")
    print("   1. Download top 30 cryptos: GET /download/top30")
    print("   2. Check database: GET /database/summary")
    print("   3. Train models: POST /train/{ticker}")
    print("   4. Get forecast: GET /forecast/{ticker}")
    print("\n💡 Tips:")
    print("   - Use /docs for interactive API testing")
    print("   - Check backend/IMPLEMENTATION_GUIDE.md for detailed documentation")
    print("   - Run backend/test_implementation.py to verify all features")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the logs in backend/app/logs/")

def main():
    """Main entry point"""
    print("=" * 60)
    print("CRYPTOCURRENCY PREDICTION PLATFORM")
    print("COM724 Assessment Implementation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Error: 'backend' directory not found")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create .env file if needed
    create_env_file()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()