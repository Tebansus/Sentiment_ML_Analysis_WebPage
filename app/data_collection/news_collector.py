import requests
from bs4 import BeautifulSoup
import datetime
from typing import List, Dict, Any
import time
from app.config.config import TRACKED_STOCKS
from app.db.database import SessionLocal
from app.db.models import NewsArticle

class NewsCollector:
    """Class for collecting financial news articles related to specified stocks"""
    
    def __init__(self):
        """Initialize news collector"""
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        self.sources = [
            {
                "name": "Yahoo Finance",
                "url_template": "https://finance.yahoo.com/quote/{symbol}/news",
                "parser": self._parse_yahoo_finance
            },
            {
                "name": "MarketWatch",
                "url_template": "https://www.marketwatch.com/investing/stock/{symbol}",
                "parser": self._parse_marketwatch
            }
        ]
    
    def _get_html_content(self, url: str) -> str:
        """Get HTML content from URL
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string
        """
        headers = {
            "User-Agent": self.user_agent
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return ""
    
    def _parse_yahoo_finance(self, html: str, source: str, symbol: str) -> List[Dict[str, Any]]:
        """Parse Yahoo Finance news articles
        
        Args:
            html: HTML content
            source: Source name
            symbol: Stock symbol
            
        Returns:
            List of article data dictionaries
        """
        articles = []
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            news_items = soup.select("div.Py\(14px\) js-stream-content Pos\(r\)")
            
            for item in news_items:
                headline_elem = item.select_one("h3")
                link_elem = item.select_one("a")
                time_elem = item.select_one("span.C\(\$c-fuji-grey-j\)")
                
                if headline_elem and link_elem and time_elem:
                    headline = headline_elem.text.strip()
                    url = "https://finance.yahoo.com" + link_elem["href"] if link_elem["href"].startswith("/") else link_elem["href"]
                    
                    # Parse time string
                    time_str = time_elem.text.strip()
                    published_at = self._parse_relative_time(time_str)
                    
                    article_data = {
                        "source": source,
                        "headline": headline,
                        "url": url,
                        "published_at": published_at,
                        "summary": ""
                    }
                    
                    articles.append(article_data)
        
        except Exception as e:
            print(f"Error parsing Yahoo Finance: {e}")
        
        return articles
    
    def _parse_marketwatch(self, html: str, source: str, symbol: str) -> List[Dict[str, Any]]:
        """Parse MarketWatch news articles
        
        Args:
            html: HTML content
            source: Source name
            symbol: Stock symbol
            
        Returns:
            List of article data dictionaries
        """
        articles = []
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            news_items = soup.select("div.article__content")
            
            for item in news_items:
                headline_elem = item.select_one("h3.article__headline")
                link_elem = headline_elem.select_one("a") if headline_elem else None
                time_elem = item.select_one("span.article__timestamp")
                
                if headline_elem and link_elem:
                    headline = headline_elem.text.strip()
                    url = link_elem["href"]
                    
                    # Parse time string
                    if time_elem:
                        time_str = time_elem.text.strip()
                        published_at = self._parse_timestamp(time_str)
                    else:
                        published_at = datetime.datetime.utcnow()
                    
                    article_data = {
                        "source": source,
                        "headline": headline,
                        "url": url,
                        "published_at": published_at,
                        "summary": ""
                    }
                    
                    articles.append(article_data)
        
        except Exception as e:
            print(f"Error parsing MarketWatch: {e}")
        
        return articles
    
    def _parse_relative_time(self, time_str: str) -> datetime.datetime:
        """Parse relative time string (e.g., '2 hours ago')
        
        Args:
            time_str: Relative time string
            
        Returns:
            Datetime object
        """
        now = datetime.datetime.utcnow()
        
        try:
            if "minute" in time_str:
                minutes = int(time_str.split()[0])
                return now - datetime.timedelta(minutes=minutes)
            elif "hour" in time_str:
                hours = int(time_str.split()[0])
                return now - datetime.timedelta(hours=hours)
            elif "day" in time_str:
                days = int(time_str.split()[0])
                return now - datetime.timedelta(days=days)
            else:
                return now
        except:
            return now
    
    def _parse_timestamp(self, time_str: str) -> datetime.datetime:
        """Parse timestamp string
        
        Args:
            time_str: Timestamp string
            
        Returns:
            Datetime object
        """
        try:
            # Try different formats
            formats = [
                "%b %d, %Y %I:%M %p ET",
                "%b %d, %Y at %I:%M %p ET",
                "%b %d, %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(time_str, fmt)
                except:
                    continue
                
            return datetime.datetime.utcnow()
        except:
            return datetime.datetime.utcnow()
    
    def collect_news(self) -> List[Dict[str, Any]]:
        """Collect news articles for tracked stocks
        
        Returns:
            List of article data dictionaries
        """
        all_articles = []
        
        for symbol in TRACKED_STOCKS:
            for source in self.sources:
                url = source["url_template"].format(symbol=symbol.lower())
                html = self._get_html_content(url)
                
                if html:
                    articles = source["parser"](html, source["name"], symbol)
                    all_articles.extend(articles)
                
                # Sleep to avoid rate limiting
                time.sleep(1)
        
        return all_articles
    
    def save_news_to_db(self, articles: List[Dict[str, Any]]) -> int:
        """Save collected news articles to database
        
        Args:
            articles: List of article data dictionaries
            
        Returns:
            Number of articles saved
        """
        db = SessionLocal()
        count = 0
        
        try:
            for article_data in articles:
                # Check if article already exists
                existing_article = db.query(NewsArticle).filter(NewsArticle.url == article_data["url"]).first()
                
                if not existing_article:
                    # Create new article
                    article = NewsArticle(**article_data)
                    db.add(article)
                    count += 1
            
            db.commit()
        
        except Exception as e:
            db.rollback()
            print(f"Error saving articles to database: {e}")
        
        finally:
            db.close()
        
        return count
    
    def run_collection(self) -> int:
        """Run full collection process
        
        Returns:
            Number of new articles collected and saved
        """
        articles = self.collect_news()
        return self.save_news_to_db(articles)


if __name__ == "__main__":
    # For testing the collector
    collector = NewsCollector()
    new_articles = collector.run_collection()
    print(f"Collected {new_articles} new articles") 