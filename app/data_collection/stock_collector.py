import yfinance as yf
import pandas as pd
import datetime
import time
import random
from typing import List, Dict, Any
from app.config.config import TRACKED_STOCKS
from app.db.database import SessionLocal
from app.db.models import Stock, StockPrice

class StockCollector:
    """Class for collecting stock market data"""
    
    def __init__(self):
        """Initialize stock collector"""
        # Company name mappings to avoid API lookups
        self.company_names = {
            "INTC": "Intel Corporation",
            "AMD": "Advanced Micro Devices, Inc.",
            "NVDA": "NVIDIA Corporation"
        }
        
        self.sectors = {
            "INTC": "Technology",
            "AMD": "Technology",
            "NVDA": "Technology"
        }
    
    def ensure_stocks_exist(self) -> None:
        """Ensure that all tracked stocks exist in the database"""
        db = SessionLocal()
        
        try:
            # Use hardcoded stock info instead of API lookups
            stock_info = {}
            for symbol in TRACKED_STOCKS:
                stock_info[symbol] = {
                    "company_name": self.company_names.get(symbol, symbol),
                    "sector": self.sectors.get(symbol, "Technology")
                }
            
            # Ensure stocks exist in database
            for symbol in TRACKED_STOCKS:
                # Check if stock already exists
                existing_stock = db.query(Stock).filter(Stock.symbol == symbol).first()
                
                if not existing_stock:
                    # Create new stock
                    stock = Stock(
                        symbol=symbol,
                        company_name=stock_info[symbol]["company_name"],
                        sector=stock_info[symbol]["sector"],
                        is_active=True
                    )
                    db.add(stock)
            
            db.commit()
        
        except Exception as e:
            db.rollback()
            print(f"Error ensuring stocks exist: {e}")
        
        finally:
            db.close()
    
    def _fetch_with_retry(self, ticker, start=None, end=None, period=None, max_retries=3):
        """Fetch stock data with retry logic
        
        Args:
            ticker: yfinance Ticker object
            start: Start date
            end: End date
            period: Time period string
            max_retries: Maximum number of retries
            
        Returns:
            Pandas DataFrame with stock data
        """
        retries = 0
        while retries < max_retries:
            try:
                if period:
                    data = ticker.history(period=period)
                else:
                    data = ticker.history(start=start, end=end)
                
                if data.empty:
                    # If data is empty but no error, handle it gracefully
                    print(f"No data returned for {ticker.ticker} (attempt {retries+1}/{max_retries})")
                    retries += 1
                    time.sleep(2 + random.random() * 3)  # Random delay between 2-5 seconds
                    continue
                
                return data
            
            except Exception as e:
                print(f"Error fetching {ticker.ticker} data (attempt {retries+1}/{max_retries}): {e}")
                retries += 1
                time.sleep(5 + random.random() * 5)  # Random delay between 5-10 seconds
        
        # If all retries failed, return empty DataFrame
        return pd.DataFrame()
    
    def collect_historical_prices(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Collect historical stock prices
        
        Args:
            days_back: Number of days to look back (reduced to avoid rate limits)
            
        Returns:
            List of price data dictionaries
        """
        # Calculate start date
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        all_prices = []
        
        for symbol in TRACKED_STOCKS:
            try:
                print(f"Collecting historical data for {symbol}...")
                # Download historical data with retry
                ticker = yf.Ticker(symbol)
                
                # Add delay to avoid rate limiting
                time.sleep(2)  # 2 second delay between requests
                
                data = self._fetch_with_retry(ticker, start=start_date, end=end_date)
                
                if data.empty:
                    print(f"No historical data available for {symbol}")
                    continue
                
                # Convert to list of dictionaries
                for index, row in data.iterrows():
                    price_data = {
                        "symbol": symbol,
                        "timestamp": index.to_pydatetime(),
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"]
                    }
                    
                    all_prices.append(price_data)
                
                print(f"Collected {len(data)} data points for {symbol}")
            
            except Exception as e:
                print(f"Error collecting historical prices for {symbol}: {e}")
            
            # Add delay between stocks
            time.sleep(3)
        
        return all_prices
    
    def collect_latest_prices(self) -> List[Dict[str, Any]]:
        """Collect latest stock prices
        
        Returns:
            List of price data dictionaries
        """
        latest_prices = []
        
        for symbol in TRACKED_STOCKS:
            try:
                print(f"Collecting latest data for {symbol}...")
                # Download latest data with retry
                ticker = yf.Ticker(symbol)
                
                # Add delay to avoid rate limiting
                time.sleep(2)  # 2 second delay between requests
                
                data = self._fetch_with_retry(ticker, period="1d")
                
                if data.empty:
                    print(f"No latest data available for {symbol}")
                    continue
                
                latest_data = data.iloc[-1]
                
                price_data = {
                    "symbol": symbol,
                    "timestamp": data.index[-1].to_pydatetime(),
                    "open": latest_data["Open"],
                    "high": latest_data["High"],
                    "low": latest_data["Low"],
                    "close": latest_data["Close"],
                    "volume": latest_data["Volume"]
                }
                
                latest_prices.append(price_data)
                print(f"Collected latest data for {symbol}")
            
            except Exception as e:
                print(f"Error collecting latest price for {symbol}: {e}")
            
            # Add delay between stocks
            time.sleep(3)
        
        return latest_prices
    
    def save_prices_to_db(self, prices: List[Dict[str, Any]]) -> int:
        """Save collected prices to database
        
        Args:
            prices: List of price data dictionaries
            
        Returns:
            Number of prices saved
        """
        if not prices:
            print("No prices to save to database")
            return 0
            
        db = SessionLocal()
        count = 0
        
        try:
            # Get stock ID mapping
            stock_mapping = {}
            stocks = db.query(Stock).all()
            for stock in stocks:
                stock_mapping[stock.symbol] = stock.stock_id
            
            for price_data in prices:
                symbol = price_data.pop("symbol")
                stock_id = stock_mapping.get(symbol)
                
                if stock_id:
                    # Check if price already exists
                    existing_price = db.query(StockPrice).filter(
                        StockPrice.stock_id == stock_id,
                        StockPrice.timestamp == price_data["timestamp"]
                    ).first()
                    
                    if not existing_price:
                        # Create new price
                        price = StockPrice(stock_id=stock_id, **price_data)
                        db.add(price)
                        count += 1
            
            db.commit()
            print(f"Saved {count} new prices to database")
        
        except Exception as e:
            db.rollback()
            print(f"Error saving prices to database: {e}")
        
        finally:
            db.close()
        
        return count
    
    def run_historical_collection(self, days_back: int = 30) -> int:
        """Run historical data collection process
        
        Args:
            days_back: Number of days to look back (reduced to avoid rate limits)
            
        Returns:
            Number of new prices collected and saved
        """
        self.ensure_stocks_exist()
        prices = self.collect_historical_prices(days_back)
        return self.save_prices_to_db(prices)
    
    def run_latest_collection(self) -> int:
        """Run latest data collection process
        
        Returns:
            Number of new prices collected and saved
        """
        self.ensure_stocks_exist()
        prices = self.collect_latest_prices()
        return self.save_prices_to_db(prices)


if __name__ == "__main__":
    # For testing the collector
    collector = StockCollector()
    
    # Collect historical data (run once for initial data)
    # new_historical = collector.run_historical_collection()
    # print(f"Collected {new_historical} historical price points")
    
    # Collect latest data (run regularly)
    new_latest = collector.run_latest_collection()
    print(f"Collected {new_latest} latest price points") 