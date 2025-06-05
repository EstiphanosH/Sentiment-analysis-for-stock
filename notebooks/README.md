# 🧪 Jupyter Notebooks for Stock Sentiment Analysis

This directory contains a collection of Jupyter Notebooks for performing sentiment analysis, exploratory data analysis (EDA), correlation checks, and time series forecasting on stock market data using financial news and stock price data.

## 📂 Notebook Descriptions

### 📘 `eda_analysis.ipynb`
Explores the structure, distribution, and initial patterns within the financial news and stock datasets.
- Missing value detection
- Distribution plots of sentiment labels/scores
- Basic ticker-wise grouping and statistics

### 💬 `sentiment_analysis.ipynb`
Performs NLP-based sentiment analysis on financial news headlines.
- Text preprocessing
- VADER/TextBlob sentiment scoring
- Sentiment classification and visualization

### 📉 `correlation_analysis.ipynb`
Analyzes correlation between sentiment scores and stock price movements.
- Merges sentiment with historical stock data
- Computes Pearson/Spearman correlations
- Visualizes relationships using heatmaps and scatter plots

### 📊 `yfance_analysis.ipynb`
Fetches and analyzes stock price data using the `yfinance` API.
- Downloads OHLCV data
- Adds technical indicators (e.g., moving averages)
- Plots historical performance for selected tickers

### ⏳ `tsa.ipynb`
Performs time series analysis on stock price data.
- Time series decomposition
- Stationarity testing (ADF/KPSS)
- Rolling statistics and forecasting prep
- Technical analysis

## 🧰 Utility Scripts

### 🛠 `create_notebook.py`
Python script to automatically generate new notebook templates with standard metadata and cell structure.

### `__init__.py`
Marks this directory as a Python package, potentially allowing import of local utilities across scripts.


## ▶️ How to Use

1. Launch Jupyter:
   ```bash
   cd notebooks
   jupyter notebook
   ```

2. Open and run the notebooks in the order:
   1. `eda_analysis.ipynb`
   2. `sentiment_analysis.ipynb`
   3. `yfance_analysis.ipynb`
   4. `correlation_analysis.ipynb`
   5. `tsa.ipynb`

## 🔧 Dependencies

Make sure you have installed the dependencies listed in the root `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📝 Notes

- Some notebooks rely on intermediate data outputs from others — follow the order above for reproducibility.
- Data files are assumed to be located in the `data/` directory at the project root.
- Outputs such as figures, processed data, or models may be saved in the `outputs/` or `reports/` directories.

---

Feel free to explore, modify, and extend the notebooks for deeper insights or new datasets.
