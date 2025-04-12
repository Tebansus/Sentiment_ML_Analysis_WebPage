import datetime
import random
import json
from typing import List, Dict
import pandas as pd
import numpy as np
from app.db.database import SessionLocal
from app.db.models import (
    Stock, 
    StockPrice, 
    Tweet, 
    NewsArticle, 
    TweetSentiment, 
    NewsSentiment,
    AggregatedSentiment
)
from app.config.config import TRACKED_STOCKS

class MockDataGenerator:
    """Generate mock data for development and testing"""
    
    def __init__(self):
        """Initialize generator"""
        self.db = SessionLocal()
        
        # Stock metadata
        self.stock_metadata = {
            "INTC": {
                "name": "Intel Corporation",
                "sector": "Technology",
                "starting_price": 45.0,
                "volatility": 0.02
            },
            "AMD": {
                "name": "Advanced Micro Devices, Inc.",
                "sector": "Technology",
                "starting_price": 120.0,
                "volatility": 0.03
            },
            "NVDA": {
                "name": "NVIDIA Corporation",
                "sector": "Technology",
                "starting_price": 400.0,
                "volatility": 0.04
            }
        }
        
        # News sources
        self.news_sources = [
            "Bloomberg", "CNBC", "Reuters", "Wall Street Journal", 
            "MarketWatch", "Financial Times", "Yahoo Finance", "Seeking Alpha"
        ]
        
        # User handles
        self.user_handles = [
            "trader123", "stockguru", "marketwatcher", "wallstreetbets", 
            "investorpro", "financenews", "stockanalyst", "marketexpert",
            "techtrader", "semiconinvestor", "chipstocks", "tradingview"
        ]
    
    def generate_stocks(self) -> List[Stock]:
        """Generate stock records
        
        Returns:
            List of generated stock objects
        """
        stocks = []
        
        # Check for existing stocks first
        existing_symbols = [s.symbol for s in self.db.query(Stock.symbol).all()]
        
        for symbol in TRACKED_STOCKS:
            # Skip if stock already exists
            if symbol in existing_symbols:
                continue
                
            # Create stock
            stock = Stock(
                symbol=symbol,
                company_name=self.stock_metadata[symbol]["name"],
                sector=self.stock_metadata[symbol]["sector"],
                is_active=True
            )
            stocks.append(stock)
        
        return stocks
    
    def generate_stock_prices(self, days: int = 60) -> Dict[str, List[StockPrice]]:
        """Generate historical stock prices
        
        Args:
            days: Number of days of historical data to generate
            
        Returns:
            Dictionary of stock symbols to price data
        """
        stock_prices = {}
        
        # Get stocks from DB
        stocks = self.db.query(Stock).all()
        stock_map = {stock.symbol: stock for stock in stocks}
        
        # Generate price data for each stock
        for symbol in TRACKED_STOCKS:
            if symbol not in stock_map:
                print(f"Stock {symbol} not found in database")
                continue
                
            stock_id = stock_map[symbol].stock_id
            
            # Parameters
            starting_price = self.stock_metadata[symbol]["starting_price"]
            volatility = self.stock_metadata[symbol]["volatility"]
            
            # Generate random walk for closing prices
            np.random.seed(42 + hash(symbol) % 100)  # Different seed for each stock
            returns = np.random.normal(0.0005, volatility, days)
            prices = [starting_price]
            
            for r in returns:
                prices.append(prices[-1] * (1 + r))
            
            # Generate daily prices
            stock_prices[symbol] = []
            end_date = datetime.datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
            
            for i in range(days):
                date = end_date - datetime.timedelta(days=days-i-1)
                
                # Skip weekends (simple approach)
                if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    continue
                
                close_price = prices[i]
                price_range = close_price * 0.02  # 2% range for high/low
                
                # Generate OHLC data
                open_price = close_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.005) * close_price)
                low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.005) * close_price)
                volume = int(np.random.normal(5000000, 2000000))
                
                # Create price record
                price = StockPrice(
                    stock_id=stock_id,
                    timestamp=date,
                    open=round(open_price, 2),
                    high=round(high_price, 2),
                    low=round(low_price, 2),
                    close=round(close_price, 2),
                    volume=max(0, volume)
                )
                
                stock_prices[symbol].append(price)
        
        return stock_prices
    
    def generate_tweets(self, count_per_stock: int = 100) -> List[Tweet]:
        """Generate mock tweets
        
        Args:
            count_per_stock: Number of tweets per stock
            
        Returns:
            List of generated tweet objects
        """
        tweets = []
        
        for symbol in TRACKED_STOCKS:
            for i in range(count_per_stock):
                # Random timestamp in last 30 days
                days_ago = random.randint(0, 30)
                hours_ago = random.randint(0, 23)
                timestamp = datetime.datetime.now() - datetime.timedelta(days=days_ago, hours=hours_ago)
                
                # Random sentiment tendency (positive, negative, or neutral)
                sentiment_tendency = random.choice(["positive", "negative", "neutral"])
                
                # Generate tweet text based on sentiment
                text = self._generate_tweet_text(symbol, sentiment_tendency)
                
                # Random metrics
                user_handle = random.choice(self.user_handles)
                user_followers = random.randint(100, 50000)
                likes = random.randint(0, 500)
                retweets = random.randint(0, 100)
                
                # Create Tweet object
                tweet = Tweet(
                    tweet_id=f"{symbol}-{i}-{int(timestamp.timestamp())}",
                    tweet_text=text,
                    user_handle=user_handle,
                    user_followers=user_followers,
                    timestamp=timestamp,
                    likes=likes,
                    retweets=retweets,
                    raw_json=json.dumps({
                        "tweet": {"id": i, "text": text},
                        "user": {"username": user_handle, "followers": user_followers}
                    })
                )
                
                tweets.append(tweet)
        
        return tweets
    
    def generate_news_articles(self, count_per_stock: int = 50) -> List[NewsArticle]:
        """Generate mock news articles
        
        Args:
            count_per_stock: Number of articles per stock
            
        Returns:
            List of generated article objects
        """
        articles = []
        
        # Get existing URLs to avoid duplicates
        existing_urls = [url[0] for url in self.db.query(NewsArticle.url).all()]
        url_counter = {}  # Track URL counters per stock
        
        for symbol in TRACKED_STOCKS:
            url_counter[symbol] = 0
            
            for i in range(count_per_stock):
                # Create unique URL with timestamp to avoid collisions
                base_url = f"https://example.com/news/{symbol}/{url_counter[symbol]}"
                while base_url in existing_urls:
                    url_counter[symbol] += 1
                    base_url = f"https://example.com/news/{symbol}/{url_counter[symbol]}"
                
                # Now we have a unique URL, track it
                existing_urls.append(base_url)
                url_counter[symbol] += 1
                
                # Random timestamp in last 30 days
                days_ago = random.randint(0, 30)
                hours_ago = random.randint(0, 23)
                timestamp = datetime.datetime.now() - datetime.timedelta(days=days_ago, hours=hours_ago)
                
                # Random sentiment tendency (positive, negative, or neutral)
                sentiment_tendency = random.choice(["positive", "negative", "neutral"])
                
                # Generate headline and summary
                headline = self._generate_headline(symbol, sentiment_tendency)
                summary = self._generate_article_summary(symbol, sentiment_tendency)
                
                # Create NewsArticle object
                article = NewsArticle(
                    source=random.choice(self.news_sources),
                    headline=headline,
                    url=base_url,
                    published_at=timestamp,
                    summary=summary
                )
                
                articles.append(article)
        
        return articles
    
    def generate_sentiments(self) -> None:
        """Generate sentiment data for tweets and news articles"""
        # Get stocks
        stocks = self.db.query(Stock).all()
        stock_map = {stock.symbol: stock for stock in stocks}
        
        # Process tweets
        tweets = self.db.query(Tweet).all()
        tweet_sentiments = []
        
        for tweet in tweets:
            # Determine which stock(s) the tweet is about
            relevant_stocks = []
            for symbol in TRACKED_STOCKS:
                if (symbol in tweet.tweet_text or 
                    self.stock_metadata[symbol]["name"] in tweet.tweet_text):
                    relevant_stocks.append(symbol)
            
            # If no relevant stock found, assign to a random one
            if not relevant_stocks:
                relevant_stocks = [random.choice(TRACKED_STOCKS)]
            
            # Generate sentiment scores based on tweet text
            sentiment_scores = self._generate_sentiment_scores(tweet.tweet_text)
            
            # Create sentiment records for each relevant stock
            for symbol in relevant_stocks:
                if symbol in stock_map:
                    sentiment = TweetSentiment(
                        tweet_id=tweet.tweet_id,
                        stock_id=stock_map[symbol].stock_id,
                        **sentiment_scores
                    )
                    tweet_sentiments.append(sentiment)
        
        # Process news articles
        articles = self.db.query(NewsArticle).all()
        news_sentiments = []
        
        for article in articles:
            # Determine which stock(s) the article is about
            relevant_stocks = []
            text = article.headline
            if article.summary:
                text += " " + article.summary
                
            for symbol in TRACKED_STOCKS:
                if (symbol in text or 
                    self.stock_metadata[symbol]["name"] in text):
                    relevant_stocks.append(symbol)
            
            # If no relevant stock found, assign to a random one
            if not relevant_stocks:
                relevant_stocks = [random.choice(TRACKED_STOCKS)]
            
            # Generate sentiment scores
            sentiment_scores = self._generate_sentiment_scores(text)
            
            # Create sentiment records for each relevant stock
            for symbol in relevant_stocks:
                if symbol in stock_map:
                    sentiment = NewsSentiment(
                        article_id=article.article_id,
                        stock_id=stock_map[symbol].stock_id,
                        **sentiment_scores
                    )
                    news_sentiments.append(sentiment)
        
        # Add sentiments to database
        try:
            self.db.add_all(tweet_sentiments)
            self.db.add_all(news_sentiments)
            self.db.commit()
            print(f"Generated {len(tweet_sentiments)} tweet sentiments and {len(news_sentiments)} news sentiments")
        except Exception as e:
            self.db.rollback()
            print(f"Error adding sentiments: {e}")
    
    def generate_aggregated_sentiments(self) -> None:
        """Generate aggregated sentiment data"""
        # Get stocks
        stocks = self.db.query(Stock).all()
        
        # Time windows
        time_windows = ["1d", "3d", "7d"]
        
        # Process each stock
        aggregated_sentiments = []
        
        for stock in stocks:
            # Get tweet sentiments
            tweet_sentiments = self.db.query(TweetSentiment).filter(
                TweetSentiment.stock_id == stock.stock_id
            ).join(
                Tweet, TweetSentiment.tweet_id == Tweet.tweet_id
            ).all()
            
            # Get news sentiments
            news_sentiments = self.db.query(NewsSentiment).filter(
                NewsSentiment.stock_id == stock.stock_id
            ).join(
                NewsArticle, NewsSentiment.article_id == NewsArticle.article_id
            ).all()
            
            # Group by day
            tweet_sentiments_by_day = {}
            for ts in tweet_sentiments:
                day = ts.tweet.timestamp.date()
                if day not in tweet_sentiments_by_day:
                    tweet_sentiments_by_day[day] = []
                tweet_sentiments_by_day[day].append(ts)
            
            news_sentiments_by_day = {}
            for ns in news_sentiments:
                day = ns.article.published_at.date()
                if day not in news_sentiments_by_day:
                    news_sentiments_by_day[day] = []
                news_sentiments_by_day[day].append(ns)
            
            # Generate aggregations for each day in the last 30 days
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=30)
            current_date = start_date
            
            while current_date <= end_date:
                # For each time window
                for window in time_windows:
                    days = int(window[0])
                    
                    # Get sentiments in the window
                    window_start = current_date - datetime.timedelta(days=days-1)
                    window_tweets = []
                    window_news = []
                    
                    for day in range(days):
                        check_date = window_start + datetime.timedelta(days=day)
                        if check_date in tweet_sentiments_by_day:
                            window_tweets.extend(tweet_sentiments_by_day[check_date])
                        if check_date in news_sentiments_by_day:
                            window_news.extend(news_sentiments_by_day[check_date])
                    
                    # Calculate aggregated metrics
                    if window_tweets or window_news:
                        tweet_sentiment = (
                            sum(ts.compound_score for ts in window_tweets) / len(window_tweets)
                            if window_tweets else None
                        )
                        news_sentiment = (
                            sum(ns.compound_score for ns in window_news) / len(window_news)
                            if window_news else None
                        )
                        
                        # Combined sentiment (60% news, 40% tweets)
                        if tweet_sentiment is not None and news_sentiment is not None:
                            combined_sentiment = 0.4 * tweet_sentiment + 0.6 * news_sentiment
                        elif tweet_sentiment is not None:
                            combined_sentiment = tweet_sentiment
                        elif news_sentiment is not None:
                            combined_sentiment = news_sentiment
                        else:
                            combined_sentiment = 0
                        
                        # Create aggregated sentiment
                        agg = AggregatedSentiment(
                            stock_id=stock.stock_id,
                            timestamp=datetime.datetime.combine(current_date, datetime.time(0, 0)),
                            time_window=window,
                            avg_tweet_sentiment=tweet_sentiment,
                            avg_news_sentiment=news_sentiment,
                            combined_sentiment=combined_sentiment,
                            tweet_volume=len(window_tweets),
                            news_volume=len(window_news)
                        )
                        
                        aggregated_sentiments.append(agg)
                
                current_date += datetime.timedelta(days=1)
        
        # Add aggregations to database
        try:
            self.db.add_all(aggregated_sentiments)
            self.db.commit()
            print(f"Generated {len(aggregated_sentiments)} aggregated sentiments")
        except Exception as e:
            self.db.rollback()
            print(f"Error adding aggregated sentiments: {e}")
    
    def _generate_tweet_text(self, symbol: str, sentiment: str) -> str:
        """Generate tweet text with the given sentiment
        
        Args:
            symbol: Stock symbol
            sentiment: "positive", "negative", or "neutral"
            
        Returns:
            Generated tweet text
        """
        company_name = self.stock_metadata[symbol]["name"]
        
        positive_templates = [
            f"Bullish on {symbol}! The company looks strong for Q2.",
            f"Just bought more {symbol} shares. Looking good!",
            f"{company_name} crushing it today! #investing",
            f"Great earnings from {symbol}, expecting a rally soon.",
            f"The new products from {company_name} are game changers!",
            f"{symbol} is a buy at these levels. Strong fundamentals.",
            f"Analyst upgrade for {symbol} - price target raised to $$$."
        ]
        
        negative_templates = [
            f"Bearish on {symbol}. Competition is getting tough.",
            f"Sold my {symbol} shares. Don't like the outlook.",
            f"{company_name} disappointing investors again. #stocks",
            f"Weak guidance from {symbol}. Expect more downside.",
            f"Supply chain issues will hurt {company_name} this quarter.",
            f"{symbol} overvalued at current levels. Time to trim.",
            f"Analyst downgrade for {symbol} - concerns about growth."
        ]
        
        neutral_templates = [
            f"Watching {symbol} closely. Could go either way.",
            f"Holding my {symbol} position for now. No changes.",
            f"{company_name} reports earnings next week. #markets",
            f"Interesting developments at {symbol}. Need more data.",
            f"Anyone have thoughts on {company_name}'s new strategy?",
            f"{symbol} trading flat today despite market volatility.",
            f"Analysis of {symbol}'s market position: competitive but challenged."
        ]
        
        if sentiment == "positive":
            return random.choice(positive_templates)
        elif sentiment == "negative":
            return random.choice(negative_templates)
        else:
            return random.choice(neutral_templates)
    
    def _generate_headline(self, symbol: str, sentiment: str) -> str:
        """Generate news headline with the given sentiment
        
        Args:
            symbol: Stock symbol
            sentiment: "positive", "negative", or "neutral"
            
        Returns:
            Generated headline
        """
        company_name = self.stock_metadata[symbol]["name"]
        
        positive_templates = [
            f"{company_name} Beats Earnings Expectations, Raises Guidance",
            f"{symbol} Shares Surge on Strong Sales Report",
            f"Analysts Bullish on {company_name} Following Product Launch",
            f"{symbol} Announces Expansion Plans, Stock Rallies",
            f"New CEO Appointment Boosts {company_name} Investor Confidence"
        ]
        
        negative_templates = [
            f"{company_name} Misses Revenue Targets, Shares Slide",
            f"{symbol} Cuts Forecast Amid Growing Competition",
            f"Investors Concerned About {company_name}'s Market Share Loss",
            f"{symbol} Faces Regulatory Scrutiny, Stock Under Pressure",
            f"Supply Chain Disruptions Impact {company_name}'s Production"
        ]
        
        neutral_templates = [
            f"{company_name} Reports In-Line Results for Q2",
            f"{symbol} to Announce Strategic Review Next Month",
            f"Market Awaits {company_name}'s Response to Industry Changes",
            f"{symbol} Maintains Current Outlook Despite Sector Volatility",
            f"Analysis: The Road Ahead for {company_name} in Changing Market"
        ]
        
        if sentiment == "positive":
            return random.choice(positive_templates)
        elif sentiment == "negative":
            return random.choice(negative_templates)
        else:
            return random.choice(neutral_templates)
    
    def _generate_article_summary(self, symbol: str, sentiment: str) -> str:
        """Generate news article summary with the given sentiment
        
        Args:
            symbol: Stock symbol
            sentiment: "positive", "negative", or "neutral"
            
        Returns:
            Generated summary
        """
        company_name = self.stock_metadata[symbol]["name"]
        
        positive_templates = [
            f"{company_name} reported quarterly earnings above analyst expectations, with revenue growing 15% year-over-year. The company also raised its full-year guidance, citing strong demand for its products and successful cost-cutting initiatives.",
            f"Shares of {symbol} climbed 8% today after the company announced a major new contract worth $500 million. Analysts at several firms upgraded the stock, pointing to improving growth prospects.",
            f"The new product lineup from {company_name} is receiving positive reviews from industry experts, who predict substantial market share gains. The stock has outperformed the sector by 12% this month."
        ]
        
        negative_templates = [
            f"{company_name} disappointed investors with quarterly results below expectations. Revenue fell 5% year-over-year, and management lowered guidance for the next quarter, citing challenging market conditions.",
            f"Shares of {symbol} dropped 6% following news of increasing competition in its core markets. An analyst report highlighted concerns about margin pressure and slowing growth.",
            f"{company_name} faces mounting challenges as new regulatory requirements could increase costs significantly. The company also announced delays in key product launches, impacting near-term outlook."
        ]
        
        neutral_templates = [
            f"{company_name} reported results in line with analyst expectations. While the company maintained its full-year guidance, management noted both opportunities and challenges in the current market environment.",
            f"Investors are closely watching {symbol} ahead of its upcoming product announcement. The stock has traded within a narrow range as the market weighs potential impacts on future growth.",
            f"Industry analysts released a mixed assessment of {company_name}'s competitive position. While acknowledging strengths in certain segments, the report also highlighted areas where the company may need to improve."
        ]
        
        if sentiment == "positive":
            return random.choice(positive_templates)
        elif sentiment == "negative":
            return random.choice(negative_templates)
        else:
            return random.choice(neutral_templates)
    
    def _generate_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Generate sentiment scores based on text content
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sentiment scores
        """
        # Simple rule-based approach for demonstration
        positive_words = ["bullish", "buy", "strong", "beats", "raising", "surge", "rally", "boost", "growth", "positive", "upgrade", "outperform"]
        negative_words = ["bearish", "sell", "weak", "misses", "cutting", "slide", "drop", "pressure", "loss", "negative", "downgrade", "underperform"]
        
        text_lower = text.lower()
        
        # Count sentiment words
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        total = len(text.split())
        
        # Calculate scores
        positive_score = min(0.95, pos_count / (total + 1) * 5)
        negative_score = min(0.95, neg_count / (total + 1) * 5)
        
        # Neutral score is remainder
        neutral_score = max(0.05, 1.0 - positive_score - negative_score)
        
        # Ensure scores sum to 1
        total_score = positive_score + negative_score + neutral_score
        positive_score /= total_score
        negative_score /= total_score
        neutral_score /= total_score
        
        # Compound score between -1 and 1
        compound_score = (positive_score - negative_score) * (1 - neutral_score * 0.5)
        
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score,
            "compound_score": compound_score
        }
    
    def generate_all_data(self) -> None:
        """Generate all mock data and populate the database"""
        print("Generating mock data...")
        
        try:
            # Generate stocks
            stocks = self.generate_stocks()
            self.db.add_all(stocks)
            self.db.commit()
            print(f"Generated {len(stocks)} stocks")
            
            # Generate stock prices
            prices_by_stock = self.generate_stock_prices()
            for symbol, prices in prices_by_stock.items():
                self.db.add_all(prices)
                print(f"Generated {len(prices)} price points for {symbol}")
            self.db.commit()
            
            # Generate tweets
            tweets = self.generate_tweets()
            self.db.add_all(tweets)
            self.db.commit()
            print(f"Generated {len(tweets)} tweets")
            
            # Generate news articles
            articles = self.generate_news_articles()
            self.db.add_all(articles)
            self.db.commit()
            print(f"Generated {len(articles)} news articles")
            
            # Generate sentiments
            self.generate_sentiments()
            
            # Generate aggregated sentiments
            self.generate_aggregated_sentiments()
            
            print("Mock data generation complete")
        
        except Exception as e:
            self.db.rollback()
            print(f"Error generating mock data: {e}")
        
        finally:
            self.db.close()


if __name__ == "__main__":
    generator = MockDataGenerator()
    generator.generate_all_data() 