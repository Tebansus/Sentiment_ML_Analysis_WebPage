import datetime
from typing import Dict, List, Tuple, Optional

from app.db.database import SessionLocal
from app.db.models import (
    Tweet, 
    NewsArticle, 
    Stock, 
    TweetSentiment, 
    NewsSentiment,
    AggregatedSentiment
)

class SentimentAggregator:
    """Class for aggregating sentiment data across different time windows"""
    
    def __init__(self):
        """Initialize sentiment aggregator"""
        pass
    
    def aggregate_all(self) -> int:
        """Aggregate sentiment data for all stocks and time windows
        
        Returns:
            Number of aggregation records created
        """
        # Common time windows for aggregation
        windows = ["1d", "3d", "7d"]
        total_count = 0
        
        for window in windows:
            days = int(window[0])
            count = self.aggregate_for_window(days)
            total_count += count
        
        return total_count
    
    def aggregate_for_window(self, days_back: int) -> int:
        """Aggregate sentiment for a specific time window
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Number of aggregation records created
        """
        db = SessionLocal()
        count = 0
        
        try:
            # Get stocks
            stocks = db.query(Stock).filter(Stock.is_active == True).all()
            
            # Calculate time window
            now = datetime.datetime.utcnow()
            start_time = now - datetime.timedelta(days=days_back)
            window_name = f"{days_back}d"
            
            # Process each stock
            for stock in stocks:
                # Get tweet sentiments for stock in time range
                tweet_sentiments = db.query(TweetSentiment).join(
                    Tweet, TweetSentiment.tweet_id == Tweet.tweet_id
                ).filter(
                    TweetSentiment.stock_id == stock.stock_id,
                    Tweet.timestamp >= start_time
                ).all()
                
                # Get news sentiments for stock in time range
                news_sentiments = db.query(NewsSentiment).join(
                    NewsArticle, NewsSentiment.article_id == NewsArticle.article_id
                ).filter(
                    NewsSentiment.stock_id == stock.stock_id,
                    NewsArticle.published_at >= start_time
                ).all()
                
                # Calculate average sentiment scores
                avg_tweet_sentiment = self._average_compound_score([ts.compound_score for ts in tweet_sentiments]) if tweet_sentiments else None
                avg_news_sentiment = self._average_compound_score([ns.compound_score for ns in news_sentiments]) if news_sentiments else None
                
                # Calculate combined sentiment (weighted average)
                combined_sentiment = self._calculate_combined_sentiment(avg_tweet_sentiment, avg_news_sentiment)
                
                # Create aggregated sentiment record
                if combined_sentiment is not None:
                    # Check if record already exists for this time window
                    existing = db.query(AggregatedSentiment).filter(
                        AggregatedSentiment.stock_id == stock.stock_id,
                        AggregatedSentiment.time_window == window_name,
                        AggregatedSentiment.timestamp >= now.date()
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.avg_tweet_sentiment = avg_tweet_sentiment
                        existing.avg_news_sentiment = avg_news_sentiment
                        existing.combined_sentiment = combined_sentiment
                        existing.tweet_volume = len(tweet_sentiments)
                        existing.news_volume = len(news_sentiments)
                        existing.timestamp = now
                    else:
                        # Create new record
                        agg = AggregatedSentiment(
                            stock_id=stock.stock_id,
                            timestamp=now,
                            time_window=window_name,
                            avg_tweet_sentiment=avg_tweet_sentiment,
                            avg_news_sentiment=avg_news_sentiment,
                            combined_sentiment=combined_sentiment,
                            tweet_volume=len(tweet_sentiments),
                            news_volume=len(news_sentiments)
                        )
                        db.add(agg)
                        count += 1
            
            db.commit()
        
        except Exception as e:
            db.rollback()
            raise Exception(f"Error aggregating sentiment: {e}")
        
        finally:
            db.close()
        
        return count
    
    def _average_compound_score(self, scores: List[float]) -> Optional[float]:
        """Calculate average compound score
        
        Args:
            scores: List of compound scores
            
        Returns:
            Average score or None if no scores
        """
        if not scores:
            return None
        
        return sum(scores) / len(scores)
    
    def _calculate_combined_sentiment(self, tweet_sentiment: Optional[float], news_sentiment: Optional[float]) -> Optional[float]:
        """Calculate combined sentiment score
        
        Args:
            tweet_sentiment: Average tweet sentiment
            news_sentiment: Average news sentiment
            
        Returns:
            Combined sentiment score
        """
        # If both are missing, return None
        if tweet_sentiment is None and news_sentiment is None:
            return None
            
        # If one is missing, return the other
        if tweet_sentiment is None:
            return news_sentiment
        if news_sentiment is None:
            return tweet_sentiment
        
        # Weight news higher than tweets (60% news, 40% tweets)
        return 0.4 * tweet_sentiment + 0.6 * news_sentiment 