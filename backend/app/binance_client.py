from binance.client import Client
from dotenv import load_dotenv
import os

# Load environment variables from ../.env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))



api_key = os.getenv("BINANCE_API")
api_secret = os.getenv("BINANCE_SECRET")

if not api_key or not api_secret:
    raise ValueError("Binance API credentials not found in .env file.")

binance_client = Client(api_key, api_secret) 
