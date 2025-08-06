"""
Ticker mapping utilities for consistent ticker symbols across platforms
"""

# Common cryptocurrency base symbols
CRYPTO_BASE_SYMBOLS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC',
    'SHIB', 'TRX', 'LINK', 'UNI', 'ATOM', 'XLM', 'XMR', 'BCH', 'LTC', 'ETC',
    'ALGO', 'VET', 'FIL', 'HBAR', 'EGLD', 'SAND', 'MANA', 'AXS', 'THETA', 'FTM',
    'NEAR', 'FLOW', 'ICP', 'APE', 'CHZ', 'QNT', 'GRT', 'LDO', 'CRV', 'AAVE'
]

# Stablecoins need special handling
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'UST']


def get_base_symbol(ticker: str) -> str:
    """
    Extract base symbol from any ticker format
    
    Args:
        ticker: Ticker in any format (BTC, BTC-USD, BTCUSDT, etc.)
        
    Returns:
        Base symbol (e.g., BTC)
    """
    ticker = ticker.upper().strip()
    
    # Check if it's already a base symbol
    if ticker in CRYPTO_BASE_SYMBOLS or ticker in STABLECOINS:
        return ticker
    
    # Remove common suffixes
    suffixes = ['USDT', '-USD', 'USD', '-USDT', 'BUSD']
    for suffix in suffixes:
        if ticker.endswith(suffix):
            base = ticker[:-len(suffix)]
            # Make sure we didn't accidentally remove part of the symbol
            if base in CRYPTO_BASE_SYMBOLS or base in STABLECOINS:
                return base
    
    # Special case for wrapped/staked tokens
    if ticker.startswith('W') and ticker[1:] in CRYPTO_BASE_SYMBOLS:
        return ticker[1:]  # WBTC -> BTC
    if ticker.startswith('ST') and ticker[2:] in CRYPTO_BASE_SYMBOLS:
        return ticker[2:]  # STETH -> ETH
        
    # If no suffix found, it might already be a base symbol
    return ticker


def format_ticker_for_source(ticker_or_base: str, source: str) -> str:
    """
    Format ticker for specific data source, ensuring no double suffixes
    
    Args:
        ticker_or_base: Base cryptocurrency symbol (e.g., BTC) or already formatted ticker
        source: Data source ('yfinance' or 'binance')
        
    Returns:
        Properly formatted ticker for the source
    """
    ticker_or_base = ticker_or_base.upper().strip()
    
    # First, extract the base symbol to avoid double suffixes
    base_symbol = get_base_symbol(ticker_or_base)
    
    # Check if the ticker is already in the correct format
    if source.lower() == 'yfinance' and ticker_or_base.endswith('-USD'):
        return ticker_or_base
    elif source.lower() == 'binance' and ticker_or_base.endswith('USDT'):
        return ticker_or_base
    
    if source.lower() == 'yfinance':
        # YFinance format: BTC-USD
        # Special handling for stablecoins
        if base_symbol in STABLECOINS:
            return f"{base_symbol}-USD"
        return f"{base_symbol}-USD"
        
    elif source.lower() == 'binance':
        # Binance format: BTCUSDT
        # Special handling for USDT itself
        if base_symbol == 'USDT':
            return 'USDTBUSD'  # USDT is traded against BUSD
        return f"{base_symbol}USDT"
        
    else:
        # Return base symbol for unknown sources
        return base_symbol


def get_common_name(ticker: str) -> str:
    """
    Get a common display name for the cryptocurrency
    
    Args:
        ticker: Any ticker format
        
    Returns:
        Common display name
    """
    base = get_base_symbol(ticker)
    
    # Common name mappings
    name_map = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'BNB': 'Binance Coin',
        'SOL': 'Solana',
        'XRP': 'Ripple',
        'ADA': 'Cardano',
        'AVAX': 'Avalanche',
        'DOGE': 'Dogecoin',
        'DOT': 'Polkadot',
        'MATIC': 'Polygon',
        'SHIB': 'Shiba Inu',
        'TRX': 'TRON',
        'LINK': 'Chainlink',
        'UNI': 'Uniswap',
        'ATOM': 'Cosmos',
        'XLM': 'Stellar',
        'XMR': 'Monero',
        'BCH': 'Bitcoin Cash',
        'LTC': 'Litecoin',
        'ETC': 'Ethereum Classic',
        'USDT': 'Tether',
        'USDC': 'USD Coin',
        'BUSD': 'Binance USD',
        'DAI': 'Dai',
    }
    
    return name_map.get(base, base)


def is_valid_ticker(ticker: str, source: str) -> bool:
    """
    Check if a ticker is valid for the given source
    
    Args:
        ticker: Ticker symbol
        source: Data source
        
    Returns:
        True if valid, False otherwise
    """
    base = get_base_symbol(ticker)
    formatted = format_ticker_for_source(base, source)
    
    # Check if the formatted ticker matches what was provided
    return ticker.upper() == formatted.upper()


def get_top_30_base_symbols():
    """
    Get the top 30 cryptocurrency base symbols by market cap
    
    Returns:
        List of base symbols
    """
    # This is a static list but could be made dynamic
    return [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'USDC', 'ADA', 'AVAX', 'DOGE', 'TRX',
        'LINK', 'DOT', 'MATIC', 'SHIB', 'DAI', 'BCH', 'LTC', 'UNI', 'LEO', 'ATOM',
        'ETC', 'XLM', 'XMR', 'TON', 'ICP', 'FIL', 'HBAR', 'APT', 'ARB', 'NEAR'
    ]
