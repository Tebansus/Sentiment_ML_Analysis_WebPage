import argparse
import logging
import time
from datetime import datetime

from app.db.init_db import init_db
from app.data_collection.stock_collector import StockCollector
from app.data_collection.twitter_collector import TwitterCollector
from app.data_collection.news_collector import NewsCollector
from app.processing.sentiment_analyzer import SentimentAnalyzer
from app.processing.sentiment_aggregator import SentimentAggregator
from app.utils.mock_data_generator import MockDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Stock Market Sentiment Analysis')
    
    # Database initialization
    parser.add_argument('--init-db', action='store_true', help='Initialize the database')
    
    # Mock data generation
    parser.add_argument('--mock-data', action='store_true', help='Use mock data instead of API calls')
    
    # Data collection options
    collection_group = parser.add_argument_group('Data Collection')
    collection_group.add_argument('--collect-stocks', action='store_true', help='Collect stock data')
    collection_group.add_argument('--collect-tweets', action='store_true', help='Collect Twitter data')
    collection_group.add_argument('--collect-news', action='store_true', help='Collect news data')
    collection_group.add_argument('--days-back', type=int, default=30, help='Number of days to look back for historical data')
    
    # Processing options
    processing_group = parser.add_argument_group('Data Processing')
    processing_group.add_argument('--analyze-sentiment', action='store_true', help='Analyze sentiment in collected data')
    processing_group.add_argument('--aggregate-sentiment', action='store_true', help='Aggregate sentiment data')
    
    # Shortcut options
    parser.add_argument('--all', action='store_true', help='Run all steps (except DB initialization)')
    parser.add_argument('--collect-all', action='store_true', help='Run all collection steps')
    parser.add_argument('--process-all', action='store_true', help='Run all processing steps')
    
    args = parser.parse_args()
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialization complete.")
    
    # Use mock data if requested
    if args.mock_data:
        logger.info("Generating mock data...")
        generator = MockDataGenerator()
        generator.generate_all_data()
        return
    
    # Shortcuts for running multiple steps
    if args.all:
        args.collect_stocks = True
        args.collect_tweets = True
        args.collect_news = True
        args.analyze_sentiment = True
        args.aggregate_sentiment = True
    
    if args.collect_all:
        args.collect_stocks = True
        args.collect_tweets = True
        args.collect_news = True
    
    if args.process_all:
        args.analyze_sentiment = True
        args.aggregate_sentiment = True
    
    # Data collection
    if args.collect_stocks:
        logger.info("Collecting stock data...")
        stock_collector = StockCollector()
        try:
            stock_collector.run_historical_collection(args.days_back)
            stock_collector.run_latest_collection()
            logger.info("Stock data collection complete.")
        except Exception as e:
            logger.error(f"Error collecting stock data: {e}")
    
    if args.collect_tweets:
        logger.info("Collecting Twitter data...")
        twitter_collector = TwitterCollector()
        try:
            twitter_collector.run_collection(args.days_back)
            logger.info("Twitter data collection complete.")
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
    
    if args.collect_news:
        logger.info("Collecting news data...")
        news_collector = NewsCollector()
        try:
            news_collector.run_collection(args.days_back)
            logger.info("News data collection complete.")
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
    
    # Data processing
    if args.analyze_sentiment:
        logger.info("Analyzing sentiment...")
        sentiment_analyzer = SentimentAnalyzer()
        try:
            # Analyze tweets
            sentiment_analyzer.analyze_tweets()
            # Analyze news
            sentiment_analyzer.analyze_news()
            logger.info("Sentiment analysis complete.")
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
    
    if args.aggregate_sentiment:
        logger.info("Aggregating sentiment data...")
        aggregator = SentimentAggregator()
        try:
            aggregator.aggregate_all()
            logger.info("Sentiment aggregation complete.")
        except Exception as e:
            logger.error(f"Error aggregating sentiment data: {e}")
    
    logger.info("All operations completed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds") 