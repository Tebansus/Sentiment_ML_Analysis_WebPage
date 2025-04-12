import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import datetime
from dateutil.relativedelta import relativedelta
import json
import os

# API base URL - Check if running in Docker or locally
API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000/api")

# Page configuration
st.set_page_config(
    page_title="Stock Market Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #2e7d32;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #d32f2f;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #757575;
        font-weight: bold;
    }
    .stat-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-up {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-down {
        color: #d32f2f;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">Stock Market Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('Analyze sentiment and predict stock price movements for tech companies.')

# Sidebar
st.sidebar.markdown('<div class="sub-header">Navigation</div>', unsafe_allow_html=True)

# Navigation options
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Stock Detail", "Sentiment Analysis", "Predictions"]
)

# Get available stocks
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stocks():
    response = requests.get(f"{API_BASE_URL}/stocks")
    if response.status_code == 200:
        return response.json()
    return []

stocks = get_stocks()
stock_symbols = [stock["symbol"] for stock in stocks]

# Date range selector for historical data
st.sidebar.markdown('<div class="sub-header">Date Range</div>', unsafe_allow_html=True)
date_range = st.sidebar.selectbox(
    "Select time period",
    ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "Max"],
    index=2
)

# Convert date range to actual dates
end_date = datetime.datetime.now().date()
if date_range == "1 Week":
    start_date = end_date - datetime.timedelta(days=7)
elif date_range == "1 Month":
    start_date = end_date - relativedelta(months=1)
elif date_range == "3 Months":
    start_date = end_date - relativedelta(months=3)
elif date_range == "6 Months":
    start_date = end_date - relativedelta(months=6)
elif date_range == "1 Year":
    start_date = end_date - relativedelta(years=1)
else:  # Max
    start_date = end_date - relativedelta(years=5)

# Format dates for API
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Sidebar - Stock selector
st.sidebar.markdown('<div class="sub-header">Stock Selection</div>', unsafe_allow_html=True)
selected_symbol = st.sidebar.selectbox("Select a stock", stock_symbols if stock_symbols else ["INTC", "AMD", "NVDA"])

# Get stock details
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_details(symbol):
    response = requests.get(f"{API_BASE_URL}/stocks/{symbol}")
    if response.status_code == 200:
        return response.json()
    return None

# Get stock prices
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_prices(symbol, start_date, end_date):
    response = requests.get(
        f"{API_BASE_URL}/stocks/{symbol}/prices",
        params={"start_date": start_date, "end_date": end_date}
    )
    if response.status_code == 200:
        return response.json()
    return []

# Get stock sentiment
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_sentiment(symbol, start_date, end_date, aggregation="1d"):
    response = requests.get(
        f"{API_BASE_URL}/stocks/{symbol}/sentiment",
        params={"start_date": start_date, "end_date": end_date, "aggregation": aggregation}
    )
    if response.status_code == 200:
        return response.json()
    return []

# Get stock predictions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_predictions(symbol, prediction_window=1):
    response = requests.get(
        f"{API_BASE_URL}/stocks/{symbol}/predictions",
        params={"prediction_window": prediction_window}
    )
    if response.status_code == 200:
        return response.json()
    return None

# Get model performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_model_performance(symbol, prediction_window=1):
    response = requests.get(
        f"{API_BASE_URL}/model/performance",
        params={"symbol": symbol, "prediction_window": prediction_window}
    )
    if response.status_code == 200:
        return response.json()
    return None

# Format sentiment score
def format_sentiment(score):
    if score > 0.2:
        return f'<span class="sentiment-positive">{score:.2f}</span>'
    elif score < -0.2:
        return f'<span class="sentiment-negative">{score:.2f}</span>'
    else:
        return f'<span class="sentiment-neutral">{score:.2f}</span>'

# Dashboard Page
if page == "Dashboard":
    st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)
    
    # Create a multi-stock view
    col1, col2, col3 = st.columns(3)
    
    stock_data = {}
    sentiment_data = {}
    prediction_data = {}
    
    # Fetch data for all stocks
    for symbol in stock_symbols:
        stock_data[symbol] = get_stock_details(symbol)
        sentiment_data[symbol] = get_stock_sentiment(symbol, start_date_str, end_date_str)
        prediction_data[symbol] = get_stock_predictions(symbol)
    
    # Display stock summary cards
    for i, symbol in enumerate(stock_symbols):
        col = [col1, col2, col3][i % 3]
        
        with col:
            st.markdown(f'<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(f"**{symbol}**: {stock_data[symbol]['company_name'] if stock_data[symbol] else 'Loading...'}")
            
            if stock_data[symbol] and stock_data[symbol]['latest_price']:
                price = stock_data[symbol]['latest_price']['close']
                st.markdown(f"**Price**: ${price:.2f}")
            
            if stock_data[symbol] and stock_data[symbol]['latest_sentiment']:
                sentiment = stock_data[symbol]['latest_sentiment']['combined_sentiment']
                st.markdown(f"**Sentiment**: {format_sentiment(sentiment)}", unsafe_allow_html=True)
            
            if prediction_data[symbol]:
                direction = prediction_data[symbol]['direction']
                confidence = prediction_data[symbol]['confidence']
                class_name = "prediction-up" if direction == "up" else "prediction-down"
                arrow = "↑" if direction == "up" else "↓"
                st.markdown(f"**Prediction**: <span class='{class_name}'>{arrow} ({confidence:.0%})</span>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparative chart
    st.markdown('<div class="sub-header">Price Comparison</div>', unsafe_allow_html=True)
    
    price_data_all = {}
    for symbol in stock_symbols:
        prices = get_stock_prices(symbol, start_date_str, end_date_str)
        if prices:
            # Normalize to percentage change from start
            df = pd.DataFrame(prices)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate percentage change from first price
            first_close = df['close'].iloc[0]
            df['change'] = (df['close'] - first_close) / first_close * 100
            
            price_data_all[symbol] = df
    
    # Create comparative chart
    fig = go.Figure()
    
    for symbol, df in price_data_all.items():
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['change'],
            mode='lines',
            name=symbol
        ))
    
    fig.update_layout(
        title="Relative Price Performance (%)",
        xaxis_title="Date",
        yaxis_title="% Change",
        legend_title="Stocks",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment comparison
    st.markdown('<div class="sub-header">Sentiment Comparison</div>', unsafe_allow_html=True)
    
    sentiment_data_all = {}
    for symbol in stock_symbols:
        sentiments = get_stock_sentiment(symbol, start_date_str, end_date_str)
        if sentiments:
            df = pd.DataFrame(sentiments)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            sentiment_data_all[symbol] = df
    
    # Create sentiment comparison chart
    fig = go.Figure()
    
    for symbol, df in sentiment_data_all.items():
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['combined_sentiment'],
            mode='lines',
            name=symbol
        ))
    
    fig.update_layout(
        title="Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        legend_title="Stocks",
        height=500
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=start_date,
        y0=0,
        x1=end_date,
        y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Stock Detail Page
elif page == "Stock Detail":
    st.markdown(f'<div class="sub-header">{selected_symbol} Detail</div>', unsafe_allow_html=True)
    
    # Get stock data
    stock_details = get_stock_details(selected_symbol)
    prices = get_stock_prices(selected_symbol, start_date_str, end_date_str)
    sentiments = get_stock_sentiment(selected_symbol, start_date_str, end_date_str)
    
    if stock_details:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Company**: {stock_details['company_name']}")
            st.markdown(f"**Sector**: {stock_details['sector']}")
        
        with col2:
            if stock_details['latest_price']:
                price = stock_details['latest_price']['close']
                st.markdown(f"**Current Price**: ${price:.2f}")
                st.markdown(f"**Volume**: {stock_details['latest_price']['volume']:,}")
        
        with col3:
            if stock_details['latest_sentiment']:
                sentiment = stock_details['latest_sentiment']['combined_sentiment']
                st.markdown(f"**Current Sentiment**: {format_sentiment(sentiment)}", unsafe_allow_html=True)
                
                tweet_vol = stock_details['latest_sentiment']['tweet_volume'] or 0
                news_vol = stock_details['latest_sentiment']['news_volume'] or 0
                st.markdown(f"**Social Volume**: {tweet_vol + news_vol:,} mentions")
    
    # Price chart with sentiment overlay
    if prices and sentiments:
        # Prepare data
        price_df = pd.DataFrame(prices)
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.sort_values('timestamp')
        
        sentiment_df = pd.DataFrame(sentiments)
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        sentiment_df = sentiment_df.sort_values('timestamp')
        
        # Create figure with secondary Y axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_df['timestamp'],
                open=price_df['open'],
                high=price_df['high'],
                low=price_df['low'],
                close=price_df['close'],
                name="Price"
            ),
            secondary_y=False
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=sentiment_df['timestamp'],
                y=sentiment_df['combined_sentiment'],
                mode='lines',
                name="Sentiment",
                line=dict(color='purple')
            ),
            secondary_y=True
        )
        
        # Layout
        fig.update_layout(
            title=f"{selected_symbol} Price and Sentiment",
            xaxis_title="Date",
            legend_title="Data",
            height=600
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        # Add zero line for sentiment
        fig.add_shape(
            type="line",
            x0=price_df['timestamp'].min(),
            y0=0,
            x1=price_df['timestamp'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            yref="y2"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=price_df['timestamp'],
                y=price_df['volume'],
                name="Volume",
                marker_color='lightblue'
            )
        )
        
        fig.update_layout(
            title=f"{selected_symbol} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation between sentiment and price change
        st.markdown('<div class="sub-header">Sentiment-Price Correlation</div>', unsafe_allow_html=True)
        
        # Merge price and sentiment data
        price_df['date'] = price_df['timestamp'].dt.date
        sentiment_df['date'] = sentiment_df['timestamp'].dt.date
        
        # Calculate daily price change
        price_df['price_change'] = price_df['close'].pct_change() * 100
        
        # Merge on date
        merged_df = pd.merge(price_df[['date', 'price_change']], 
                             sentiment_df[['date', 'combined_sentiment']], 
                             on='date', how='inner')
        
        # Create scatter plot
        fig = px.scatter(
            merged_df, 
            x='combined_sentiment', 
            y='price_change',
            trendline='ols',
            title=f"{selected_symbol} Sentiment vs. Price Change",
            labels={
                'combined_sentiment': 'Sentiment Score',
                'price_change': 'Daily Price Change (%)'
            }
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = merged_df['combined_sentiment'].corr(merged_df['price_change'])
        st.markdown(f"**Correlation coefficient**: {correlation:.2f}")
        
        if correlation > 0.5:
            st.markdown("There is a **strong positive correlation** between sentiment and price changes.")
        elif correlation > 0.3:
            st.markdown("There is a **moderate positive correlation** between sentiment and price changes.")
        elif correlation > 0:
            st.markdown("There is a **weak positive correlation** between sentiment and price changes.")
        elif correlation > -0.3:
            st.markdown("There is a **weak negative correlation** between sentiment and price changes.")
        elif correlation > -0.5:
            st.markdown("There is a **moderate negative correlation** between sentiment and price changes.")
        else:
            st.markdown("There is a **strong negative correlation** between sentiment and price changes.")

# Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.markdown(f'<div class="sub-header">{selected_symbol} Sentiment Analysis</div>', unsafe_allow_html=True)
    
    # Get sentiment data with different aggregation windows
    sentiments_1d = get_stock_sentiment(selected_symbol, start_date_str, end_date_str, "1d")
    sentiments_3d = get_stock_sentiment(selected_symbol, start_date_str, end_date_str, "3d")
    sentiments_7d = get_stock_sentiment(selected_symbol, start_date_str, end_date_str, "7d")
    
    # Convert to DataFrames
    df_1d = pd.DataFrame(sentiments_1d)
    df_3d = pd.DataFrame(sentiments_3d)
    df_7d = pd.DataFrame(sentiments_7d)
    
    if not df_1d.empty:
        df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
        df_1d = df_1d.sort_values('timestamp')
        
        # Sentiment trend chart
        st.markdown('<div class="sub-header">Sentiment Trend</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Add sentiment components
        fig.add_trace(go.Scatter(
            x=df_1d['timestamp'],
            y=df_1d['tweet_sentiment'],
            mode='lines',
            name='Twitter Sentiment'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_1d['timestamp'],
            y=df_1d['news_sentiment'],
            mode='lines',
            name='News Sentiment'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_1d['timestamp'],
            y=df_1d['combined_sentiment'],
            mode='lines',
            name='Combined Sentiment',
            line=dict(width=3)
        ))
        
        # Layout
        fig.update_layout(
            title="Sentiment Components Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            legend_title="Source",
            height=500
        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=df_1d['timestamp'].min(),
            y0=0,
            x1=df_1d['timestamp'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment volume
        st.markdown('<div class="sub-header">Social Media & News Volume</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_1d['timestamp'],
            y=df_1d['tweet_volume'],
            name='Twitter Volume'
        ))
        
        fig.add_trace(go.Bar(
            x=df_1d['timestamp'],
            y=df_1d['news_volume'],
            name='News Volume'
        ))
        
        # Layout
        fig.update_layout(
            title="Social Media & News Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Different time window comparison
        st.markdown('<div class="sub-header">Sentiment Time Windows</div>', unsafe_allow_html=True)
        
        # Prepare data
        if not df_3d.empty and not df_7d.empty:
            df_3d['timestamp'] = pd.to_datetime(df_3d['timestamp'])
            df_3d = df_3d.sort_values('timestamp')
            
            df_7d['timestamp'] = pd.to_datetime(df_7d['timestamp'])
            df_7d = df_7d.sort_values('timestamp')
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_1d['timestamp'],
                y=df_1d['combined_sentiment'],
                mode='lines',
                name='1-Day Window'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_3d['timestamp'],
                y=df_3d['combined_sentiment'],
                mode='lines',
                name='3-Day Window'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_7d['timestamp'],
                y=df_7d['combined_sentiment'],
                mode='lines',
                name='7-Day Window'
            ))
            
            # Layout
            fig.update_layout(
                title="Sentiment with Different Time Windows",
                xaxis_title="Date",
                yaxis_title="Combined Sentiment Score",
                legend_title="Time Window",
                height=400
            )
            
            # Add zero line
            fig.add_shape(
                type="line",
                x0=df_1d['timestamp'].min(),
                y0=0,
                x1=df_1d['timestamp'].max(),
                y1=0,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Note on Time Windows**:
            - **1-Day**: Shows daily sentiment, more reactive to short-term events
            - **3-Day**: Smooths out daily fluctuations, shows short-term trends
            - **7-Day**: Shows broader sentiment trends, less affected by daily noise
            """)

# Predictions Page
elif page == "Predictions":
    st.markdown(f'<div class="sub-header">{selected_symbol} Price Predictions</div>', unsafe_allow_html=True)
    
    # Prediction window selector
    prediction_window = st.radio(
        "Prediction timeframe",
        [1, 3, 7],
        horizontal=True,
        format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
    )
    
    # Get prediction data
    prediction = get_stock_predictions(selected_symbol, prediction_window)
    model_performance = get_model_performance(selected_symbol, prediction_window)
    
    if prediction:
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(f"### Prediction for {prediction_window} day{'s' if prediction_window > 1 else ''}")
            
            direction = prediction['direction']
            confidence = prediction['confidence']
            price_change = prediction['predicted_price_change'] * 100
            
            direction_text = "Bullish 📈" if direction == "up" else "Bearish 📉"
            direction_class = "prediction-up" if direction == "up" else "prediction-down"
            
            st.markdown(f"**Direction**: <span class='{direction_class}'>{direction_text}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence**: {confidence:.1%}")
            st.markdown(f"**Predicted Change**: {price_change:.2f}%")
            
            st.markdown(f"**Current Price**: ${prediction['current_price']:.2f}")
            st.markdown(f"**Predicted Price**: ${prediction['predicted_price']:.2f}")
            
            st.markdown(f"**Prediction Made**: {prediction['timestamp']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if model_performance and "accuracy" in model_performance:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("### Model Performance")
                
                accuracy = model_performance['accuracy']
                avg_confidence = model_performance['average_confidence']
                total_predictions = model_performance['total_predictions']
                
                # Accuracy gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=accuracy * 100,
                    title={'text': "Historical Accuracy"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=250)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**Average Confidence**: {avg_confidence:.1%}")
                st.markdown(f"**Based on**: {total_predictions} predictions")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Get prices for backtesting chart
    prices = get_stock_prices(selected_symbol, start_date_str, end_date_str)
    
    if prices:
        # Convert to DataFrame
        price_df = pd.DataFrame(prices)
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.sort_values('timestamp')
        
        # Calculate future values for backtesting
        price_df[f'future_{prediction_window}d'] = price_df['close'].shift(-prediction_window)
        price_df[f'actual_change_{prediction_window}d'] = (price_df[f'future_{prediction_window}d'] - price_df['close']) / price_df['close'] * 100
        
        # Remove rows with NaN values (at the end)
        price_df = price_df.dropna()
        
        # Create backtesting chart
        st.markdown(f"### Backtesting {prediction_window}-day Price Changes")
        
        fig = go.Figure()
        
        # Add closing price
        fig.add_trace(go.Scatter(
            x=price_df['timestamp'],
            y=price_df['close'],
            mode='lines',
            name='Close Price'
        ))
        
        # Add secondary y-axis for percentage change
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add closing price on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=price_df['timestamp'],
                y=price_df['close'],
                mode='lines',
                name='Close Price'
            ),
            secondary_y=False
        )
        
        # Add actual change on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=price_df['timestamp'],
                y=price_df[f'actual_change_{prediction_window}d'],
                name=f'Actual {prediction_window}-day Change (%)',
                marker_color=price_df[f'actual_change_{prediction_window}d'].apply(
                    lambda x: 'green' if x > 0 else 'red'
                )
            ),
            secondary_y=True
        )
        
        # Layout
        fig.update_layout(
            title=f"{selected_symbol} Price with {prediction_window}-day Future Changes",
            xaxis_title="Date",
            legend_title="Data",
            height=600
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="% Change", secondary_y=True)
        
        # Add zero line for percent change
        fig.add_shape(
            type="line",
            x0=price_df['timestamp'].min(),
            y0=0,
            x1=price_df['timestamp'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            yref="y2"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of price changes
        st.markdown(f"### Distribution of {prediction_window}-day Price Changes")
        
        fig = px.histogram(
            price_df, 
            x=f'actual_change_{prediction_window}d',
            nbins=30,
            title=f"Distribution of {prediction_window}-day Price Changes (%)"
        )
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # Add mean line
        mean_change = price_df[f'actual_change_{prediction_window}d'].mean()
        fig.add_vline(x=mean_change, line_dash="solid", line_color="blue",
                    annotation_text=f"Mean: {mean_change:.2f}%", 
                    annotation_position="top right")
        
        fig.update_layout(
            xaxis_title="Price Change (%)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats about price changes
        positive_days = (price_df[f'actual_change_{prediction_window}d'] > 0).sum()
        total_days = len(price_df)
        positive_percentage = positive_days / total_days * 100
        
        st.markdown(f"**Positive changes**: {positive_days} out of {total_days} days ({positive_percentage:.1f}%)")
        st.markdown(f"**Average change**: {price_df[f'actual_change_{prediction_window}d'].mean():.2f}%")
        st.markdown(f"**Max gain**: {price_df[f'actual_change_{prediction_window}d'].max():.2f}%")
        st.markdown(f"**Max loss**: {price_df[f'actual_change_{prediction_window}d'].min():.2f}%")

# Footer
st.markdown("---")
st.markdown("Stock Market Sentiment Analysis | Data updates every hour | ©2025")

# Add a refresh button
if st.button("Refresh Data"):
    st.experimental_rerun() 