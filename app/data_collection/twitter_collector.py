import tweepy
import json
import datetime
import time
from typing import List, Dict, Any
from app.config.config import (
    TWITTER_API_KEY, 
    TWITTER_API_SECRET, 
    TWITTER_BEARER_TOKEN,
    TRACKED_STOCKS
)
from app.db.database import SessionLocal
from app.db.models import Tweet, Stock

class TwitterCollector:
    """Class for collecting tweets related to specified stocks"""
    
    def __init__(self):
        """Initialize Twitter API client"""
        self.client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            wait_on_rate_limit=True
        )
        
    def _build_search_query(self, stock_symbols: List[str]) -> str:
        """Build search query for multiple stock symbols
        
        Args:
            stock_symbols: List of stock symbols to search for
            
        Returns:
            String query for Twitter API
        """
        # Don't use cashtags as they require premium API access
        # Just use company names and symbols without $
        query_parts = []
        for symbol in stock_symbols:
            # Add symbol without $ sign
            query_parts.append(symbol)
            
            # Add common variations of company names based on symbol
            if symbol == "INTC":
                query_parts.append("Intel")
                query_parts.append("Intel stock")
            elif symbol == "AMD":
                query_parts.append("AMD")
                query_parts.append("AMD stock")
                query_parts.append("Advanced Micro Devices")
            elif symbol == "NVDA":
                query_parts.append("Nvidia")
                query_parts.append("Nvidia stock")
        
        # Join with OR operator
        query = " OR ".join([f'"{part}"' for part in query_parts])
        
        # Add filters for better quality tweets
        query += " -is:retweet lang:en"
        
        # Make sure it doesn't exceed Twitter API query limits (512 characters)
        if len(query) > 500:
            # Simplify query if too long
            basic_parts = []
            for symbol in stock_symbols:
                basic_parts.append(symbol)
                if symbol == "INTC":
                    basic_parts.append("Intel")
                elif symbol == "AMD":
                    basic_parts.append("AMD")
                elif symbol == "NVDA":
                    basic_parts.append("Nvidia")
            
            query = " OR ".join([f'"{part}"' for part in basic_parts])
            query += " -is:retweet lang:en"
        
        print(f"Twitter search query: {query}")
        return query
    
    def collect_tweets(self, days_back: int = 7, max_results: int = 10) -> List[Dict[str, Any]]:
        """Collect tweets related to tracked stocks
        
        Args:
            days_back: Number of days to look back
            max_results: Maximum number of results to retrieve (reduced to avoid rate limits)
            
        Returns:
            List of tweet data dictionaries
        """
        # Calculate start time
        start_time = datetime.datetime.utcnow() - datetime.timedelta(days=days_back)
        
        # Build search query
        query = self._build_search_query(TRACKED_STOCKS)
        
        # Collect tweets
        tweets = []
        
        try:
            print(f"Searching for tweets with query: {query}")
            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                tweet_fields=["created_at", "public_metrics", "author_id", "text"],
                user_fields=["username", "public_metrics"],
                expansions=["author_id"]
            )
            
            # Process tweets
            if response.data:
                users = {user.id: user for user in response.includes["users"]}
                
                for tweet in response.data:
                    user = users[tweet.author_id]
                    
                    # Simplified JSON representation to avoid serialization issues
                    tweet_dict = {"id": tweet.id, "text": tweet.text}
                    user_dict = {"username": user.username, "followers": user.public_metrics["followers_count"]}
                    
                    tweet_data = {
                        "tweet_id": str(tweet.id),  # Convert to string for safer storage
                        "tweet_text": tweet.text,
                        "user_handle": user.username,
                        "user_followers": user.public_metrics["followers_count"],
                        "timestamp": tweet.created_at,
                        "likes": tweet.public_metrics["like_count"],
                        "retweets": tweet.public_metrics["retweet_count"],
                        "raw_json": json.dumps({
                            "tweet": tweet_dict,
                            "user": user_dict
                        })
                    }
                    
                    tweets.append(tweet_data)
                
                print(f"Found {len(tweets)} tweets")
            else:
                print("No tweets found matching the query")
        
        except Exception as e:
            print(f"Error collecting tweets: {e}")
        
        return tweets
    
    def save_tweets_to_db(self, tweets: List[Dict[str, Any]]) -> int:
        """Save collected tweets to database
        
        Args:
            tweets: List of tweet data dictionaries
            
        Returns:
            Number of tweets saved
        """
        if not tweets:
            print("No tweets to save to database")
            return 0
            
        db = SessionLocal()
        count = 0
        
        try:
            for tweet_data in tweets:
                # Check if tweet already exists
                existing_tweet = db.query(Tweet).filter(Tweet.tweet_id == tweet_data["tweet_id"]).first()
                
                if not existing_tweet:
                    # Create new tweet
                    tweet = Tweet(**tweet_data)
                    db.add(tweet)
                    count += 1
            
            db.commit()
            print(f"Saved {count} new tweets to database")
        
        except Exception as e:
            db.rollback()
            print(f"Error saving tweets to database: {e}")
        
        finally:
            db.close()
        
        return count
    
    def run_collection(self, days_back: int = 3, max_results: int = 10) -> int:
        """Run full collection process
        
        Args:
            days_back: Number of days to look back (reduced to avoid rate limits)
            max_results: Maximum number of results to retrieve (reduced to avoid rate limits)
            
        Returns:
            Number of new tweets collected and saved
        """
        tweets = self.collect_tweets(days_back, max_results)
        return self.save_tweets_to_db(tweets)


if __name__ == "__main__":
    # For testing the collector
    collector = TwitterCollector()
    new_tweets = collector.run_collection()
    print(f"Collected {new_tweets} new tweets") 