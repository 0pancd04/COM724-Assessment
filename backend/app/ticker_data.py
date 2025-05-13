import json
from app.binance_client import binance_client

def get_symbol_list():
    exchange_info = binance_client.get_exchange_info()

    # Extract all trading symbols
    all_symbols = [
        symbol["symbol"]
        for symbol in exchange_info.get("symbols", [])
        if symbol.get("status") == "TRADING"  # Only include actively trading pairs
    ]

    # Save to JSON file
    output_file = "data/binance_symbols.json"
    with open(output_file, "w") as f:
        json.dump(all_symbols, f, indent=4)

    print(f"Saved {len(all_symbols)} trading symbols to {output_file}")
