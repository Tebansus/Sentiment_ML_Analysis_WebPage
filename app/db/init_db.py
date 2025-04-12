import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db.database import init_db, SessionLocal
from app.db.models import Stock
from app.config.config import TRACKED_STOCKS

def init_database():
    """Initialize the database with tables and initial data"""
    print("Initializing database...")
    
    # Create tables
    init_db()
    print("Database tables created.")
    
    # Add initial stocks
    db = SessionLocal()
    try:
        # Check if stocks already exist
        existing_stocks = db.query(Stock).all()
        existing_symbols = [stock.symbol for stock in existing_stocks]
        
        # Add tracked stocks if they don't exist
        for symbol in TRACKED_STOCKS:
            if symbol not in existing_symbols:
                # Default values, will be updated by stock collector
                stock = Stock(
                    symbol=symbol,
                    company_name=get_company_name(symbol),
                    sector="Technology",
                    is_active=True
                )
                db.add(stock)
                print(f"Added stock: {symbol}")
        
        db.commit()
        print("Initial stocks added.")
    
    except Exception as e:
        db.rollback()
        print(f"Error adding initial stocks: {e}")
    
    finally:
        db.close()
    
    print("Database initialization complete.")

def get_company_name(symbol):
    """Get company name for stock symbol"""
    company_names = {
        "INTC": "Intel Corporation",
        "AMD": "Advanced Micro Devices, Inc.",
        "NVDA": "NVIDIA Corporation"
    }
    return company_names.get(symbol, symbol)

if __name__ == "__main__":
    init_database()