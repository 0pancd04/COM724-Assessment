"""
RSS Feed Handler for Cryptocurrency News
Fetches and manages news feeds from various crypto news sources
"""

import feedparser
import requests
import socket
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from .logger import setup_logger
from .analysis_storage import analysis_storage

# Setup logging
logger = setup_logger("rss_feed_handler", "logs/rss_feed.log")

class RSSFeedHandler:
    """Handles RSS feeds for cryptocurrency news"""
    
    def __init__(self):
        # Popular cryptocurrency news RSS feeds - Updated URLs
        self.feed_sources = {
            "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "cointelegraph": "https://cointelegraph.com/rss",
            "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
            "cryptonews": "https://cryptonews.com/news/feed/",
            "newsbtc": "https://www.newsbtc.com/feed/",
            "cryptoslate": "https://cryptoslate.com/feed/",
            "theblock": "https://www.theblockcrypto.com/rss.xml",
            "coinjournal": "https://coinjournal.net/feed/"
        }
        
        # Ticker keyword mapping for filtering
        self.ticker_keywords = {
            "BTC": ["bitcoin", "btc", "₿"],
            "ETH": ["ethereum", "eth", "ether"],
            "ADA": ["cardano", "ada"],
            "BNB": ["binance", "bnb"],
            "XRP": ["ripple", "xrp"],
            "SOL": ["solana", "sol"],
            "DOT": ["polkadot", "dot"],
            "DOGE": ["dogecoin", "doge"],
            "AVAX": ["avalanche", "avax"],
            "MATIC": ["polygon", "matic"],
            "LINK": ["chainlink", "link"],
            "UNI": ["uniswap", "uni"],
            "ATOM": ["cosmos", "atom"],
            "LTC": ["litecoin", "ltc"],
            "FIL": ["filecoin", "fil"]
        }
    
    def fetch_feed(self, source_name: str, source_url: str, limit: int = 10) -> List[Dict]:
        """Fetch RSS feed from a single source with timeout and error handling"""
        try:
            logger.info(f"Fetching RSS feed from {source_name}")
            
            # Add timeout and user agent to avoid blocking
            import socket
            socket.setdefaulttimeout(10)
            
            # Parse the feed with custom headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Try to fetch with requests first for better error handling
            try:
                response = requests.get(source_url, headers=headers, timeout=10)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except:
                # Fallback to direct feedparser
                feed = feedparser.parse(source_url)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed parsing issue for {source_name}: {feed.bozo_exception}")
                # Try to continue anyway if we have entries
                if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                    return []
            
            articles = []
            entries = feed.entries if hasattr(feed, 'entries') else []
            
            for entry in entries[:limit]:
                try:
                    article = {
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "description": entry.get("summary", entry.get("description", "")),
                        "published": self._parse_date(entry.get("published_parsed")),
                        "source": source_name,
                        "tags": entry.get("tags", [])
                    }
                    
                    # Determine related tickers
                    article["related_tickers"] = self._extract_tickers(
                        f"{article['title']} {article['description']}"
                    )
                    
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing entry from {source_name}: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching feed from {source_name}: {e}")
            return []
        finally:
            # Reset socket timeout
            socket.setdefaulttimeout(None)
    
    def fetch_all_feeds(self, limit_per_source: int = 5) -> List[Dict]:
        """Fetch feeds from all sources"""
        all_articles = []
        
        for source_name, source_url in self.feed_sources.items():
            articles = self.fetch_feed(source_name, source_url, limit_per_source)
            all_articles.extend(articles)
        
        # Sort by published date
        all_articles.sort(key=lambda x: x.get("published", ""), reverse=True)
        
        # Store in database
        for article in all_articles:
            # Get first related ticker or None
            ticker = article["related_tickers"][0] if article["related_tickers"] else None
            
            analysis_storage.store_rss_feed(
                ticker=ticker,
                title=article["title"],
                link=article["link"],
                description=article["description"],
                published_date=article["published"],
                source=article["source"]
            )
        
        return all_articles
    
    def fetch_ticker_news(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch news related to a specific ticker"""
        all_articles = self.fetch_all_feeds(limit_per_source=10)
        
        # Filter articles related to the ticker
        ticker_articles = []
        for article in all_articles:
            if ticker in article["related_tickers"]:
                ticker_articles.append(article)
            
            if len(ticker_articles) >= limit:
                break
        
        return ticker_articles
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract cryptocurrency tickers mentioned in text"""
        text_lower = text.lower()
        mentioned_tickers = []
        
        for ticker, keywords in self.ticker_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if ticker not in mentioned_tickers:
                        mentioned_tickers.append(ticker)
                    break
        
        return mentioned_tickers
    
    def _parse_date(self, time_struct) -> str:
        """Parse date from feedparser time struct"""
        if time_struct:
            dt = datetime.fromtimestamp(feedparser._parse_date(time_struct))
            return dt.isoformat()
        return datetime.now().isoformat()
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis for news text
        Returns a score between -1 (negative) and 1 (positive)
        """
        # Simple keyword-based sentiment (can be enhanced with ML models)
        positive_words = [
            "bullish", "surge", "rally", "gain", "rise", "growth", "positive",
            "breakthrough", "adoption", "success", "upgrade", "moon", "pump"
        ]
        
        negative_words = [
            "bearish", "crash", "fall", "drop", "decline", "negative", "concern",
            "risk", "warning", "scam", "hack", "loss", "dump", "fear"
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return sentiment
    
    def get_market_sentiment(self, ticker: str = None) -> Dict:
        """Get overall market sentiment from recent news"""
        # Get recent news from database
        recent_news = analysis_storage.get_rss_feeds(ticker, limit=50)
        
        if not recent_news:
            # Fetch fresh news if none in database
            if ticker:
                recent_news = self.fetch_ticker_news(ticker)
            else:
                recent_news = self.fetch_all_feeds()
        
        sentiments = []
        for article in recent_news:
            sentiment = self.analyze_sentiment(
                f"{article.get('title', '')} {article.get('description', '')}"
            )
            sentiments.append(sentiment)
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Determine market mood
            if avg_sentiment > 0.3:
                mood = "Very Bullish"
            elif avg_sentiment > 0.1:
                mood = "Bullish"
            elif avg_sentiment > -0.1:
                mood = "Neutral"
            elif avg_sentiment > -0.3:
                mood = "Bearish"
            else:
                mood = "Very Bearish"
            
            return {
                "sentiment_score": avg_sentiment,
                "market_mood": mood,
                "articles_analyzed": len(sentiments),
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "sentiment_score": 0.0,
            "market_mood": "Unknown",
            "articles_analyzed": 0,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }

# Initialize RSS feed handler
rss_handler = RSSFeedHandler()
