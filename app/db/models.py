from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Stock(Base):
    """Stock information table"""
    __tablename__ = "stocks"

    stock_id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True)
    company_name = Column(String(100), nullable=False)
    sector = Column(String(50))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock")
    tweet_sentiments = relationship("TweetSentiment", back_populates="stock")
    news_sentiments = relationship("NewsSentiment", back_populates="stock")
    agg_sentiments = relationship("AggregatedSentiment", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")
    
    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', company_name='{self.company_name}')>"
    
class StockPrice(Base):
    """Historical stock price data"""
    __tablename__ = "stock_prices"
    
    price_id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.stock_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")
    
    def __repr__(self):
        return f"<StockPrice(symbol='{self.stock.symbol}', timestamp='{self.timestamp}', close={self.close})>"

class Tweet(Base):
    """Collected tweets data"""
    __tablename__ = "tweets"
    
    tweet_id = Column(String(50), primary_key=True)
    tweet_text = Column(Text, nullable=False)
    user_handle = Column(String(50))
    user_followers = Column(Integer)
    timestamp = Column(DateTime, nullable=False)
    likes = Column(Integer)
    retweets = Column(Integer)
    raw_json = Column(JSON)
    
    # Relationships
    sentiments = relationship("TweetSentiment", back_populates="tweet")
    
    def __repr__(self):
        return f"<Tweet(id='{self.tweet_id}', user='{self.user_handle}')>"

class NewsArticle(Base):
    """News articles data"""
    __tablename__ = "news_articles"
    
    article_id = Column(Integer, primary_key=True)
    source = Column(String(100), nullable=False)
    headline = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False, unique=True)
    published_at = Column(DateTime, nullable=False)
    summary = Column(Text)
    
    # Relationships
    sentiments = relationship("NewsSentiment", back_populates="article")
    
    def __repr__(self):
        return f"<NewsArticle(id={self.article_id}, source='{self.source}')>"

class TweetSentiment(Base):
    """Sentiment analysis results for tweets"""
    __tablename__ = "tweet_sentiments"
    
    sentiment_id = Column(Integer, primary_key=True)
    tweet_id = Column(String(50), ForeignKey("tweets.tweet_id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.stock_id"), nullable=False)
    positive_score = Column(Float, nullable=False)
    negative_score = Column(Float, nullable=False)
    neutral_score = Column(Float, nullable=False)
    compound_score = Column(Float, nullable=False)
    
    # Relationships
    tweet = relationship("Tweet", back_populates="sentiments")
    stock = relationship("Stock", back_populates="tweet_sentiments")
    
    def __repr__(self):
        return f"<TweetSentiment(tweet_id='{self.tweet_id}', stock='{self.stock.symbol}', compound={self.compound_score})>"

class NewsSentiment(Base):
    """Sentiment analysis results for news articles"""
    __tablename__ = "news_sentiments"
    
    sentiment_id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("news_articles.article_id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.stock_id"), nullable=False)
    positive_score = Column(Float, nullable=False)
    negative_score = Column(Float, nullable=False)
    neutral_score = Column(Float, nullable=False)
    compound_score = Column(Float, nullable=False)
    
    # Relationships
    article = relationship("NewsArticle", back_populates="sentiments")
    stock = relationship("Stock", back_populates="news_sentiments")
    
    def __repr__(self):
        return f"<NewsSentiment(article_id={self.article_id}, stock='{self.stock.symbol}', compound={self.compound_score})>"

class AggregatedSentiment(Base):
    """Aggregated sentiment metrics"""
    __tablename__ = "aggregated_sentiments"
    
    agg_id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.stock_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    time_window = Column(String(20), nullable=False)  # e.g., "1d", "7d"
    avg_tweet_sentiment = Column(Float)
    avg_news_sentiment = Column(Float)
    combined_sentiment = Column(Float)
    tweet_volume = Column(Integer)
    news_volume = Column(Integer)
    
    # Relationships
    stock = relationship("Stock", back_populates="agg_sentiments")
    
    def __repr__(self):
        return f"<AggregatedSentiment(stock='{self.stock.symbol}', timestamp='{self.timestamp}', combined={self.combined_sentiment})>"

class Prediction(Base):
    """Price movement predictions"""
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.stock_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    prediction_window = Column(Integer, nullable=False)  # Days ahead
    predicted_price_change = Column(Float, nullable=False)
    prediction_confidence = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(stock='{self.stock.symbol}', window={self.prediction_window}d, confidence={self.prediction_confidence})>" 