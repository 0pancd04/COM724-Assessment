from binance.client import Client
from dotenv import load_dotenv
import os

# Load environment variables from ../.env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))



api_key = os.getenv("BINANCE_API")
api_secret = os.getenv("BINANCE_SECRET")

# Create a dummy client if credentials are not available
if not api_key or not api_secret:
    print("Warning: Binance API credentials not found. Using dummy client.")
    binance_client = None
else:
    binance_client = Client(api_key, api_secret) 
