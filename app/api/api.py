from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import datetime

from app.db.database import get_db
from app.db.models import Stock, StockPrice, AggregatedSentiment, Prediction
from app.models.lstm_model import LSTMModel
from app.config.config import MODEL_VERSION, PREDICTION_WINDOWS

app = FastAPI(
    title="Stock Market Sentiment Analysis API",
    description="API for stock market sentiment analysis and price prediction",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = LSTMModel()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Stock Market Sentiment Analysis API"}

@app.get("/api/stocks")
def get_stocks(db: Session = Depends(get_db)):
    """Get all tracked stocks"""
    stocks = db.query(Stock).filter(Stock.is_active == True).all()
    
    return [
        {
            "stock_id": stock.stock_id,
            "symbol": stock.symbol,
            "company_name": stock.company_name,
            "sector": stock.sector
        }
        for stock in stocks
    ]

@app.get("/api/stocks/{symbol}")
def get_stock_details(symbol: str, db: Session = Depends(get_db)):
    """Get stock details"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Get latest price
    latest_price = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.stock_id
    ).order_by(StockPrice.timestamp.desc()).first()
    
    # Get latest sentiment
    latest_sentiment = db.query(AggregatedSentiment).filter(
        AggregatedSentiment.stock_id == stock.stock_id,
        AggregatedSentiment.time_window == "1d"
    ).order_by(AggregatedSentiment.timestamp.desc()).first()
    
    return {
        "stock_id": stock.stock_id,
        "symbol": stock.symbol,
        "company_name": stock.company_name,
        "sector": stock.sector,
        "latest_price": {
            "timestamp": latest_price.timestamp if latest_price else None,
            "open": latest_price.open if latest_price else None,
            "high": latest_price.high if latest_price else None,
            "low": latest_price.low if latest_price else None,
            "close": latest_price.close if latest_price else None,
            "volume": latest_price.volume if latest_price else None
        } if latest_price else None,
        "latest_sentiment": {
            "timestamp": latest_sentiment.timestamp if latest_sentiment else None,
            "tweet_sentiment": latest_sentiment.avg_tweet_sentiment if latest_sentiment else None,
            "news_sentiment": latest_sentiment.avg_news_sentiment if latest_sentiment else None,
            "combined_sentiment": latest_sentiment.combined_sentiment if latest_sentiment else None,
            "tweet_volume": latest_sentiment.tweet_volume if latest_sentiment else None,
            "news_volume": latest_sentiment.news_volume if latest_sentiment else None
        } if latest_sentiment else None
    }

@app.get("/api/stocks/{symbol}/prices")
def get_stock_prices(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    interval: str = "1d",
    db: Session = Depends(get_db)
):
    """Get historical stock prices"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Parse dates
    start_datetime = None
    end_datetime = None
    
    if start_date:
        try:
            start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    
    if end_date:
        try:
            end_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
    
    # Build query
    query = db.query(StockPrice).filter(StockPrice.stock_id == stock.stock_id)
    
    if start_datetime:
        query = query.filter(StockPrice.timestamp >= start_datetime)
    
    if end_datetime:
        query = query.filter(StockPrice.timestamp <= end_datetime)
    
    # Order by timestamp
    prices = query.order_by(StockPrice.timestamp.asc()).all()
    
    return [
        {
            "timestamp": price.timestamp,
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume
        }
        for price in prices
    ]

@app.get("/api/stocks/{symbol}/sentiment")
def get_stock_sentiment(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    aggregation: str = "1d",
    db: Session = Depends(get_db)
):
    """Get stock sentiment analysis"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Parse dates
    start_datetime = None
    end_datetime = None
    
    if start_date:
        try:
            start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    
    if end_date:
        try:
            end_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
    
    # Validate aggregation window
    valid_windows = ["1d", "3d", "7d"]
    if aggregation not in valid_windows:
        aggregation = "1d"
    
    # Build query
    query = db.query(AggregatedSentiment).filter(
        AggregatedSentiment.stock_id == stock.stock_id,
        AggregatedSentiment.time_window == aggregation
    )
    
    if start_datetime:
        query = query.filter(AggregatedSentiment.timestamp >= start_datetime)
    
    if end_datetime:
        query = query.filter(AggregatedSentiment.timestamp <= end_datetime)
    
    # Order by timestamp
    sentiments = query.order_by(AggregatedSentiment.timestamp.asc()).all()
    
    return [
        {
            "timestamp": sentiment.timestamp,
            "time_window": sentiment.time_window,
            "tweet_sentiment": sentiment.avg_tweet_sentiment,
            "news_sentiment": sentiment.avg_news_sentiment,
            "combined_sentiment": sentiment.combined_sentiment,
            "tweet_volume": sentiment.tweet_volume,
            "news_volume": sentiment.news_volume
        }
        for sentiment in sentiments
    ]

@app.get("/api/stocks/{symbol}/predictions")
def get_stock_predictions(
    symbol: str, 
    prediction_window: int = Query(1, description="Prediction window in days"),
    db: Session = Depends(get_db)
):
    """Get stock price predictions"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Validate prediction window
    if prediction_window not in PREDICTION_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Invalid prediction_window. Use one of {PREDICTION_WINDOWS}")
    
    # Get latest prediction from database
    latest_prediction = db.query(Prediction).filter(
        Prediction.stock_id == stock.stock_id,
        Prediction.prediction_window == prediction_window
    ).order_by(Prediction.timestamp.desc()).first()
    
    # If no prediction exists or it's older than 24 hours, generate a new one
    if not latest_prediction or (datetime.datetime.utcnow() - latest_prediction.timestamp).total_seconds() > 86400:
        prediction_result = model.predict(stock.stock_id, prediction_window)
        
        if not prediction_result:
            raise HTTPException(status_code=404, detail="Could not generate prediction")
        
        return prediction_result
    
    # Get latest price for context
    latest_price = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.stock_id
    ).order_by(StockPrice.timestamp.desc()).first()
    
    current_price = latest_price.close if latest_price else 0
    predicted_price = current_price * (1 + latest_prediction.predicted_price_change)
    
    return {
        "stock_id": stock.stock_id,
        "symbol": stock.symbol,
        "prediction_window": latest_prediction.prediction_window,
        "current_price": current_price,
        "predicted_price_change": latest_prediction.predicted_price_change,
        "predicted_price": predicted_price,
        "direction": "up" if latest_prediction.predicted_price_change > 0 else "down",
        "confidence": latest_prediction.prediction_confidence,
        "timestamp": latest_prediction.timestamp,
        "model_version": latest_prediction.model_version
    }

@app.post("/api/stocks")
def add_stock(symbol: str, db: Session = Depends(get_db)):
    """Add stock to track"""
    # Check if stock already exists
    existing = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if existing:
        if not existing.is_active:
            # Reactivate stock
            existing.is_active = True
            db.commit()
            return {"message": f"Stock {symbol} reactivated"}
        else:
            return {"message": f"Stock {symbol} is already tracked"}
    
    # Import here to avoid circular import
    from app.data_collection.stock_collector import StockCollector
    
    # Initialize collector
    collector = StockCollector()
    
    # Ensure stock exists
    try:
        collector.ensure_stocks_exist()
        return {"message": f"Stock {symbol} added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/stocks/{symbol}")
def remove_stock(symbol: str, db: Session = Depends(get_db)):
    """Remove stock from tracking"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Mark as inactive instead of deleting
    stock.is_active = False
    db.commit()
    
    return {"message": f"Stock {symbol} removed from tracking"}

@app.post("/api/model/train")
def train_model(
    symbol: str = Query(..., description="Stock symbol to train model for"),
    prediction_window: int = Query(1, description="Prediction window in days"),
    db: Session = Depends(get_db)
):
    """Trigger model retraining"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Validate prediction window
    if prediction_window not in PREDICTION_WINDOWS:
        raise HTTPException(status_code=400, detail=f"Invalid prediction_window. Use one of {PREDICTION_WINDOWS}")
    
    try:
        # Train model
        model.model = None  # Reset model
        model.price_scaler = None
        model.sentiment_scaler = None
        model.feature_scaler = None
        
        metrics = model.train(stock.stock_id, prediction_window)
        
        return {
            "message": f"Model trained successfully for {symbol} with {prediction_window}-day window",
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/performance")
def get_model_performance(
    symbol: str = Query(..., description="Stock symbol"),
    prediction_window: int = Query(1, description="Prediction window in days"),
    db: Session = Depends(get_db)
):
    """Get model performance metrics"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Get all predictions for this stock and window
    predictions = db.query(Prediction).filter(
        Prediction.stock_id == stock.stock_id,
        Prediction.prediction_window == prediction_window
    ).order_by(Prediction.timestamp.desc()).limit(30).all()
    
    if not predictions:
        return {"message": "No predictions available"}
    
    # Calculate accuracy (simplified)
    # In a real implementation, you'd compare predictions with actual outcomes
    correct_predictions = sum(1 for p in predictions if p.predicted_price_change > 0)
    accuracy = correct_predictions / len(predictions)
    
    return {
        "symbol": symbol,
        "prediction_window": prediction_window,
        "model_version": MODEL_VERSION,
        "total_predictions": len(predictions),
        "accuracy": accuracy,
        "average_confidence": sum(p.prediction_confidence for p in predictions) / len(predictions),
        "latest_prediction_time": predictions[0].timestamp if predictions else None,
    } 