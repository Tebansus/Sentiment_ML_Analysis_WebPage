# Stock Market Sentiment Analysis

A comprehensive tool for analyzing sentiment in social media and news about stocks to aid investment decisions. Currently working with mock data, looking for alternative API for Twitter.

## Project Overview

This application collects data from various sources including stock prices, tweets, and news articles about tracked stocks. It then analyzes the sentiment of tweets and news to provide insights about market sentiment and how it might relate to stock price movements.

## Features

- **Data Collection**:
  - Stock price data from Yahoo Finance 
  - Tweets related to tracked stocks
  - News articles about tracked stocks

- **Sentiment Analysis**:
  - Analyze sentiment in tweets and news using NLTK VADER
  - Aggregate sentiment across different time windows
  - Visualize sentiment trends over time

- **Dashboard**:
  - Interactive Streamlit dashboard to visualize data
  - Stock price charts with sentiment overlay
  - Sentiment trends and statistics

## Technical Architecture

```
SentimentAnalysisProject/
├── app/                    # Main application code
│   ├── api/                # API endpoints (FastAPI)
│   ├── config/             # Configuration settings
│   ├── data_collection/    # Data collection modules
│   ├── db/                 # Database models and connections
│   ├── frontend/           # Streamlit dashboard
│   ├── models/             # ML models for prediction
│   ├── processing/         # Data processing modules
│   └── utils/              # Utility functions
├── data/                   # Data storage
│   ├── interim/            # Intermediate data
│   ├── processed/          # Processed data
│   └── raw/                # Raw collected data
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Test suite
├── main.py                 # Command-line entry point
├── run.py                  # FastAPI application entry point
└── requirements.txt        # Dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL
- Twitter API credentials
- API keys for financial data sources

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-sentiment-analysis.git
cd stock-sentiment-analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_sentiment
DB_USER=postgres
DB_PASSWORD=your_password

# Twitter API
TWITTER_BEARER_TOKEN=your_token

# Other API keys as needed
```

4. Initialize the database:
```bash
python main.py --init-db
```

## Usage

### Data Collection

To collect all data:
```bash
python main.py --collect-all
```

Individual collection tasks:
```bash
python main.py --collect-stocks  # Collect stock prices
python main.py --collect-tweets  # Collect tweets
python main.py --collect-news    # Collect news articles
```

### Sentiment Analysis

Process sentiment:
```bash
python main.py --process-all
```

### Run Everything

To run all collection and processing:
```bash
python main.py --all
```

### Using Mock Data

For development or testing without API access:
```bash
python main.py --mock-data
```

This generates realistic mock data for all components including:
- Stock price data with realistic market behavior
- Tweets with sentiment tendencies
- News articles with varied sentiment
- Sentiment analysis results

### Running the API

Start the FastAPI server:
```bash
python run.py
```

### Dashboard

Start the Streamlit dashboard:
```bash
streamlit run app/frontend/app.py
```

## Docker Deployment

Using Docker Compose:

```bash
docker-compose up -d
docker-compose exec backend python main.py --init-db
docker-compose exec backend python main.py --all

# Or using mock data
docker-compose exec backend python main.py --mock-data
```

Access the dashboard at http://localhost:8501

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and VADER Sentiment Analysis
- TensorFlow and Keras
- Twitter API
- FastAPI and Streamlit teams

---

*Disclaimer: This system is for educational purposes only and should not be used for actual investment decisions.* 
