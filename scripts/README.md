# ⚙️ Python Scripts for Stock Sentiment Analysis

This directory contains modular Python scripts that power the sentiment analysis, financial data processing, correlation computation, and visualization for the project.

Each script is designed to be reusable and focused on a specific task within the pipeline of stock sentiment analysis and forecasting.

---

## 📜 Script Descriptions

### `correlation_analysis.py`
- Computes correlation between sentiment scores and stock price changes.
- Supports Pearson and Spearman correlation types.
- Can output correlation heatmaps and individual ticker analysis.

### `sentiment_analysis.py`
- Contains a `SentimentAnalyzer` class for processing and scoring financial news headlines.
- Utilizes NLP tools like VADER or TextBlob for sentiment scoring.
- Methods for cleaning, preprocessing, classifying, and visualizing sentiment trends.

### `technical_analysis.py`
- Provides functionality for calculating common technical indicators using TA-Lib or custom logic.
- Indicators include RSI, MACD, moving averages, etc.
- Plots technical signals over historical stock price charts.

### `time_series_analysis.py`
- Contains the `TimeSeriesAnalyzer` class for decomposing and analyzing stock time series.
- Supports stationarity tests (ADF, KPSS), seasonal decomposition, and trend smoothing.
- Prepares data for forecasting models.

### `utils.py`
- Utility functions for loading data, formatting plots, date parsing, and common transformations.
- Used across multiple scripts and notebooks to reduce code redundancy.

---

## ▶️ How to Use

You can import classes and functions from these scripts into your own Python files or Jupyter notebooks:

```python
from scripts.sentiment_analysis import SentimentAnalyzer
from scripts.correlation_analysis import CorrelationAnalyzer
from scripts.technical_analysis import TechnicalAnalyzer
from scripts.time_series_analysis import TimeSeriesAnalyzer
```

Each class includes modular methods to:
- Load and prepare data
- Perform specific analyses
- Generate interactive plots

---

## 📦 Dependencies

Make sure the project dependencies are installed via:

```bash
pip install -r ../requirements.txt
```

---

## 🧪 Testing

Scripts are designed to support unit testing using `pytest`. Tests are typically placed in the `tests/` directory.

---

## 📝 Notes

- Ensure your data files (news and stock data) are formatted properly before passing to any analyzer classes.
- Customize visual outputs using Plotly themes and layout settings in `utils.py`.

---

Happy analyzing! 📈🧠
