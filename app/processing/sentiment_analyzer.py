import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import datetime
from typing import Dict, Any, List, Tuple
from app.db.database import SessionLocal
from app.db.models import (
    Tweet, 
    NewsArticle, 
    Stock, 
    TweetSentiment, 
    NewsSentiment,
    AggregatedSentiment
)

# Download NLTK resources (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """Class for analyzing sentiment in financial text"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Financial terms augmentation with custom sentiment scores
        self.financial_lexicon = {
            # Positive terms
            "beat": 3.0,
            "exceeds": 2.0,
            "growth": 1.5,
            "growing": 1.5,
            "bullish": 2.5,
            "upgrade": 2.0,
            "upgraded": 2.0,
            "buy": 1.5,
            "strong buy": 2.5,
            "outperform": 2.0,
            "overweight": 2.0,
            "profit": 1.5,
            "profitable": 1.5,
            "earnings beat": 3.0,
            "dividend": 1.0,
            "raised guidance": 2.5,
            "raised forecast": 2.5,
            "momentum": 1.0,
            "new high": 2.0,
            "all-time high": 2.5,
            
            # Negative terms
            "miss": -3.0,
            "misses": -3.0,
            "bearish": -2.5,
            "downgrade": -2.0,
            "downgraded": -2.0,
            "sell": -1.5,
            "strong sell": -2.5,
            "underperform": -2.0,
            "underweight": -2.0,
            "loss": -1.5,
            "earnings miss": -3.0,
            "cut dividend": -2.0,
            "lowered guidance": -2.5,
            "lowered forecast": -2.5,
            "new low": -2.0,
            "52-week low": -2.5,
            "downtrend": -1.5,
            "bankruptcy": -3.5,
            "lawsuit": -2.0,
            "investigation": -2.0,
            "scandal": -2.5,
            "recession": -2.0,
            "volatility": -1.0,
            "debt": -1.0,
            "inflation": -1.0,
            "layoffs": -2.0,
            "restructuring": -1.0
        }
        
        # Augment VADER lexicon with financial terms
        self.analyzer.lexicon.update(self.financial_lexicon)
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.analyzer.polarity_scores(text)
        
        return {
            "positive_score": scores["pos"],
            "negative_score": scores["neg"],
            "neutral_score": scores["neu"],
            "compound_score": scores["compound"]
        }
    
    def process_new_tweets(self) -> int:
        """Process new tweets without sentiment analysis
        
        Returns:
            Number of tweets processed
        """
        db = SessionLocal()
        count = 0
        
        try:
            # Get stocks
            stocks = db.query(Stock).filter(Stock.is_active == True).all()
            stock_dict = {stock.symbol: stock.stock_id for stock in stocks}
            
            # Get tweets without sentiment analysis
            processed_tweet_ids = db.query(TweetSentiment.tweet_id).distinct().all()
            processed_tweet_ids = [t[0] for t in processed_tweet_ids]
            
            tweets = db.query(Tweet).filter(Tweet.tweet_id.notin_(processed_tweet_ids)).all()
            
            for tweet in tweets:
                # Analyze sentiment
                sentiment_scores = self.analyze_text(tweet.tweet_text)
                
                # Create sentiment records for each relevant stock
                for symbol, stock_id in stock_dict.items():
                    # Check if tweet contains stock symbol or name
                    if self._is_relevant_to_stock(tweet.tweet_text, symbol):
                        sentiment = TweetSentiment(
                            tweet_id=tweet.tweet_id,
                            stock_id=stock_id,
                            **sentiment_scores
                        )
                        db.add(sentiment)
                        count += 1
            
            db.commit()
        
        except Exception as e:
            db.rollback()
            print(f"Error processing tweets: {e}")
        
        finally:
            db.close()
        
        return count
    
    def process_new_articles(self) -> int:
        """Process new news articles without sentiment analysis
        
        Returns:
            Number of articles processed
        """
        db = SessionLocal()
        count = 0
        
        try:
            # Get stocks
            stocks = db.query(Stock).filter(Stock.is_active == True).all()
            stock_dict = {stock.symbol: stock.stock_id for stock in stocks}
            
            # Get articles without sentiment analysis
            processed_article_ids = db.query(NewsSentiment.article_id).distinct().all()
            processed_article_ids = [a[0] for a in processed_article_ids]
            
            articles = db.query(NewsArticle).filter(NewsArticle.article_id.notin_(processed_article_ids)).all()
            
            for article in articles:
                # Combine headline and summary for analysis
                text = article.headline
                if article.summary:
                    text += " " + article.summary
                
                # Analyze sentiment
                sentiment_scores = self.analyze_text(text)
                
                # Create sentiment records for each relevant stock
                for symbol, stock_id in stock_dict.items():
                    # Check if article is relevant to stock
                    if self._is_relevant_to_stock(text, symbol):
                        sentiment = NewsSentiment(
                            article_id=article.article_id,
                            stock_id=stock_id,
                            **sentiment_scores
                        )
                        db.add(sentiment)
                        count += 1
            
            db.commit()
        
        except Exception as e:
            db.rollback()
            print(f"Error processing articles: {e}")
        
        finally:
            db.close()
        
        return count
    
    def _is_relevant_to_stock(self, text: str, symbol: str) -> bool:
        """Check if text is relevant to a stock
        
        Args:
            text: Text to check
            symbol: Stock symbol
            
        Returns:
            True if relevant, False otherwise
        """
        text_lower = text.lower()
        
        # Check for symbol (with and without $)
        if symbol.lower() in text_lower or f"${symbol.lower()}" in text_lower:
            return True
        
        # Check for company name
        if symbol == "INTC" and "intel" in text_lower:
            return True
        elif symbol == "AMD" and ("amd" in text_lower or "advanced micro" in text_lower):
            return True
        elif symbol == "NVDA" and "nvidia" in text_lower:
            return True
        
        return False
    
    def calculate_aggregated_sentiment(self, days_back: int = 7) -> int:
        """Calculate aggregated sentiment metrics
        
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
            
            # Calculate time windows
            now = datetime.datetime.utcnow()
            time_ranges = {
                "1d": now - datetime.timedelta(days=1),
                "3d": now - datetime.timedelta(days=3),
                "7d": now - datetime.timedelta(days=7),
            }
            
            # Process each stock
            for stock in stocks:
                for window_name, start_time in time_ranges.items():
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
            print(f"Error calculating aggregated sentiment: {e}")
        
        finally:
            db.close()
        
        return count
    
    def _average_compound_score(self, scores: List[float]) -> float:
        """Calculate average compound score
        
        Args:
            scores: List of compound scores
            
        Returns:
            Average score
        """
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def _calculate_combined_sentiment(self, tweet_sentiment: float, news_sentiment: float) -> float:
        """Calculate combined sentiment score
        
        Args:
            tweet_sentiment: Average tweet sentiment
            news_sentiment: Average news sentiment
            
        Returns:
            Combined sentiment score
        """
        # If one is missing, return the other
        if tweet_sentiment is None:
            return news_sentiment
        if news_sentiment is None:
            return tweet_sentiment
        
        # Weight news higher than tweets (60% news, 40% tweets)
        return 0.4 * tweet_sentiment + 0.6 * news_sentiment
    
    def run_sentiment_processing(self) -> Tuple[int, int, int]:
        """Run full sentiment processing pipeline
        
        Returns:
            Tuple of (tweets processed, articles processed, aggregations created)
        """
        tweets_count = self.process_new_tweets()
        articles_count = self.process_new_articles()
        agg_count = self.calculate_aggregated_sentiment()
        
        return tweets_count, articles_count, agg_count


if __name__ == "__main__":
    # For testing the analyzer
    analyzer = SentimentAnalyzer()
    tweets, articles, aggs = analyzer.run_sentiment_processing()
    print(f"Processed {tweets} tweets, {articles} articles, created {aggs} aggregations") 