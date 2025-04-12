import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import mlflow
import mlflow.keras
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import joblib

from app.config.config import MODEL_DIR, MODEL_VERSION, SEQUENCE_LENGTH, PREDICTION_WINDOWS
from app.db.database import SessionLocal
from app.db.models import Stock, StockPrice, AggregatedSentiment, Prediction

class LSTMModel:
    """LSTM model for stock price prediction"""
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        """Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences (lookback window)
        """
        self.sequence_length = sequence_length
        self.model = None
        self.price_scaler = None
        self.sentiment_scaler = None
        self.feature_scaler = None
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Keras Sequential model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for price prediction
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def _prepare_data(self, stock_id: int, prediction_window: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training
        
        Args:
            stock_id: Stock ID to prepare data for
            prediction_window: Number of days ahead to predict
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        db = SessionLocal()
        
        try:
            # Get stock prices
            prices = db.query(StockPrice).filter(
                StockPrice.stock_id == stock_id
            ).order_by(StockPrice.timestamp.asc()).all()
            
            # Create price dataframe
            price_data = pd.DataFrame([
                {
                    'timestamp': p.timestamp,
                    'open': p.open,
                    'high': p.high,
                    'low': p.low,
                    'close': p.close,
                    'volume': p.volume
                } for p in prices
            ])
            
            if price_data.empty:
                raise ValueError(f"No price data found for stock ID {stock_id}")
            
            # Set timestamp as index
            price_data.set_index('timestamp', inplace=True)
            
            # Get sentiment data
            sentiments = db.query(AggregatedSentiment).filter(
                AggregatedSentiment.stock_id == stock_id,
                AggregatedSentiment.time_window == "1d"
            ).order_by(AggregatedSentiment.timestamp.asc()).all()
            
            # Create sentiment dataframe
            sentiment_data = pd.DataFrame([
                {
                    'timestamp': s.timestamp,
                    'tweet_sentiment': s.avg_tweet_sentiment if s.avg_tweet_sentiment is not None else 0,
                    'news_sentiment': s.avg_news_sentiment if s.avg_news_sentiment is not None else 0,
                    'combined_sentiment': s.combined_sentiment,
                    'tweet_volume': s.tweet_volume,
                    'news_volume': s.news_volume
                } for s in sentiments
            ])
            
            # If sentiment data is empty, create dummy data with zeros
            if sentiment_data.empty:
                sentiment_data = pd.DataFrame([
                    {
                        'timestamp': ts,
                        'tweet_sentiment': 0,
                        'news_sentiment': 0,
                        'combined_sentiment': 0,
                        'tweet_volume': 0,
                        'news_volume': 0
                    } for ts in price_data.index
                ])
            
            # Set timestamp as index
            sentiment_data.set_index('timestamp', inplace=True)
            
            # Merge price and sentiment data
            data = price_data.join(sentiment_data, how='left')
            
            # Fill missing sentiment values with zeros
            data.fillna(0, inplace=True)
            
            # Create target variable (future price change percentage)
            data['future_price'] = data['close'].shift(-prediction_window)
            data['price_change'] = (data['future_price'] - data['close']) / data['close']
            
            # Drop rows with NaN (at the end due to shifting)
            data.dropna(inplace=True)
            
            # Create feature set
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'tweet_sentiment', 'news_sentiment', 'combined_sentiment',
                'tweet_volume', 'news_volume'
            ]
            
            X = data[features].values
            y = data['price_change'].values
            
            # Scale features
            if self.price_scaler is None:
                self.price_scaler = MinMaxScaler()
                price_features = X[:, :5]  # price-related features
                self.price_scaler.fit(price_features)
            
            if self.sentiment_scaler is None:
                self.sentiment_scaler = MinMaxScaler(feature_range=(-1, 1))
                sentiment_features = X[:, 5:8]  # sentiment score features
                self.sentiment_scaler.fit(sentiment_features)
            
            if self.feature_scaler is None:
                self.feature_scaler = MinMaxScaler()
                volume_features = X[:, 8:]  # volume features
                self.feature_scaler.fit(volume_features)
            
            # Apply scaling
            price_features = self.price_scaler.transform(X[:, :5])
            sentiment_features = self.sentiment_scaler.transform(X[:, 5:8])
            volume_features = self.feature_scaler.transform(X[:, 8:])
            
            # Combine scaled features
            X_scaled = np.hstack((price_features, sentiment_features, volume_features))
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y)
            
            # Split into train and test sets (80% train, 20% test)
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            return X_train, y_train, X_test, y_test
        
        finally:
            db.close()
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, stock_id: int, prediction_window: int = 1, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """Train LSTM model
        
        Args:
            stock_id: Stock ID to train model for
            prediction_window: Number of days ahead to predict
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self._prepare_data(stock_id, prediction_window)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self._build_model(input_shape)
        
        # Define callbacks
        model_path = os.path.join(MODEL_DIR, f"stock_{stock_id}_window_{prediction_window}_{MODEL_VERSION}.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
        ]
        
        # Start MLflow tracking
        mlflow.set_experiment(f"Stock Prediction - Stock {stock_id}")
        
        with mlflow.start_run(run_name=f"LSTM_Window_{prediction_window}_{MODEL_VERSION}"):
            # Log parameters
            mlflow.log_param("sequence_length", self.sequence_length)
            mlflow.log_param("prediction_window", prediction_window)
            mlflow.log_param("stock_id", stock_id)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("initial_lr", 0.001)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_loss = self.model.evaluate(X_test, y_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_pred.flatten() - y_test))
            mse = np.mean(np.square(y_pred.flatten() - y_test))
            rmse = np.sqrt(mse)
            
            # Log metrics
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            
            # Log model
            mlflow.keras.log_model(self.model, "model")
            
            # Save scalers
            scaler_dir = os.path.join(MODEL_DIR, f"scalers_stock_{stock_id}_window_{prediction_window}_{MODEL_VERSION}")
            os.makedirs(scaler_dir, exist_ok=True)
            
            joblib.dump(self.price_scaler, os.path.join(scaler_dir, "price_scaler.pkl"))
            joblib.dump(self.sentiment_scaler, os.path.join(scaler_dir, "sentiment_scaler.pkl"))
            joblib.dump(self.feature_scaler, os.path.join(scaler_dir, "feature_scaler.pkl"))
            
            return {
                "loss": test_loss,
                "mae": mae,
                "mse": mse,
                "rmse": rmse
            }
    
    def load(self, stock_id: int, prediction_window: int = 1, version: str = MODEL_VERSION) -> bool:
        """Load trained model
        
        Args:
            stock_id: Stock ID the model was trained for
            prediction_window: Prediction window the model was trained for
            version: Model version to load
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        model_path = os.path.join(MODEL_DIR, f"stock_{stock_id}_window_{prediction_window}_{version}.h5")
        scaler_dir = os.path.join(MODEL_DIR, f"scalers_stock_{stock_id}_window_{prediction_window}_{version}")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_dir):
            return False
        
        try:
            # Load model
            self.model = load_model(model_path)
            
            # Load scalers
            self.price_scaler = joblib.load(os.path.join(scaler_dir, "price_scaler.pkl"))
            self.sentiment_scaler = joblib.load(os.path.join(scaler_dir, "sentiment_scaler.pkl"))
            self.feature_scaler = joblib.load(os.path.join(scaler_dir, "feature_scaler.pkl"))
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, stock_id: int, prediction_window: int = 1, confidence_threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """Make price change prediction
        
        Args:
            stock_id: Stock ID to predict for
            prediction_window: Number of days ahead to predict
            confidence_threshold: Threshold for prediction confidence
            
        Returns:
            Prediction dictionary or None if prediction couldn't be made
        """
        # Try to load model if not already loaded
        if self.model is None:
            if not self.load(stock_id, prediction_window):
                return None
        
        db = SessionLocal()
        
        try:
            # Get latest data
            prices = db.query(StockPrice).filter(
                StockPrice.stock_id == stock_id
            ).order_by(StockPrice.timestamp.desc()).limit(self.sequence_length).all()
            
            # Reverse to get chronological order
            prices = prices[::-1]
            
            if len(prices) < self.sequence_length:
                return None
            
            # Create features array
            price_features = np.array([
                [p.open, p.high, p.low, p.close, p.volume] for p in prices
            ])
            
            # Get sentiment data for the same time period
            sentiments = []
            for p in prices:
                # Get sentiment closest to this price date
                sentiment = db.query(AggregatedSentiment).filter(
                    AggregatedSentiment.stock_id == stock_id,
                    AggregatedSentiment.time_window == "1d",
                    AggregatedSentiment.timestamp <= p.timestamp
                ).order_by(AggregatedSentiment.timestamp.desc()).first()
                
                if sentiment:
                    sentiments.append([
                        sentiment.avg_tweet_sentiment if sentiment.avg_tweet_sentiment is not None else 0,
                        sentiment.avg_news_sentiment if sentiment.avg_news_sentiment is not None else 0,
                        sentiment.combined_sentiment,
                        sentiment.tweet_volume,
                        sentiment.news_volume
                    ])
                else:
                    sentiments.append([0, 0, 0, 0, 0])
            
            sentiment_features = np.array(sentiments)
            
            # Scale features
            scaled_price = self.price_scaler.transform(price_features)
            scaled_sentiment = self.sentiment_scaler.transform(sentiment_features[:, :3])
            scaled_volume = self.feature_scaler.transform(sentiment_features[:, 3:])
            
            # Combine scaled features
            X = np.hstack((scaled_price, scaled_sentiment, scaled_volume))
            
            # Reshape for LSTM input [samples, time steps, features]
            X = X.reshape(1, X.shape[0], X.shape[1])
            
            # Make prediction
            prediction = self.model.predict(X)[0][0]
            
            # Get latest stock price
            latest_price = prices[-1].close
            
            # Calculate predicted future price
            predicted_price = latest_price * (1 + prediction)
            
            # Calculate prediction confidence (simplified version)
            # In a real-world model, you'd use more sophisticated methods
            confidence = 0.7  # Placeholder
            
            # Store prediction in database
            if confidence >= confidence_threshold:
                new_prediction = Prediction(
                    stock_id=stock_id,
                    prediction_window=prediction_window,
                    predicted_price_change=float(prediction),
                    prediction_confidence=float(confidence),
                    model_version=MODEL_VERSION
                )
                db.add(new_prediction)
                db.commit()
                
                # Get stock symbol
                stock = db.query(Stock).filter(Stock.stock_id == stock_id).first()
                symbol = stock.symbol if stock else "Unknown"
                
                return {
                    "stock_id": stock_id,
                    "symbol": symbol,
                    "prediction_window": prediction_window,
                    "current_price": latest_price,
                    "predicted_price_change": float(prediction),
                    "predicted_price": float(predicted_price),
                    "direction": "up" if prediction > 0 else "down",
                    "confidence": float(confidence),
                    "timestamp": datetime.datetime.utcnow(),
                    "model_version": MODEL_VERSION
                }
            
            return None
        
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
        
        finally:
            db.close()
    
    def train_all_models(self, epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train models for all stocks and prediction windows
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary of training results
        """
        db = SessionLocal()
        results = {}
        
        try:
            # Get all active stocks
            stocks = db.query(Stock).filter(Stock.is_active == True).all()
            
            for stock in stocks:
                stock_results = {}
                
                for window in PREDICTION_WINDOWS:
                    print(f"Training model for {stock.symbol} with {window}-day prediction window")
                    
                    try:
                        # Reset model and scalers
                        self.model = None
                        self.price_scaler = None
                        self.sentiment_scaler = None
                        self.feature_scaler = None
                        
                        # Train model
                        metrics = self.train(stock.stock_id, window, epochs, batch_size)
                        stock_results[f"window_{window}"] = metrics
                        
                        print(f"Training complete: {metrics}")
                    
                    except Exception as e:
                        print(f"Error training model for {stock.symbol} with {window}-day window: {e}")
                        stock_results[f"window_{window}"] = {"error": str(e)}
                
                results[stock.symbol] = stock_results
        
        finally:
            db.close()
        
        return results


if __name__ == "__main__":
    # For testing the model
    model = LSTMModel()
    
    # Train models for all stocks and prediction windows
    # results = model.train_all_models()
    # print(results)
    
    # Make prediction for a specific stock and window
    prediction = model.predict(1, 1)  # Stock ID 1, 1-day window
    if prediction:
        print(f"Prediction: {prediction['direction']} with {prediction['confidence']:.2f} confidence")
        print(f"Predicted price change: {prediction['predicted_price_change']:.2%}")
        print(f"Current price: ${prediction['current_price']:.2f}, Predicted: ${prediction['predicted_price']:.2f}") 