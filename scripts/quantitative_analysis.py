import pandas as pd
import talib as ta
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class QuantitativeAnalyzer:
    def __init__(self, price_data: pd.DataFrame, ticker: str):
        """
        Analyze technical indicators for a single ticker.
        
        Args:
            price_data: DataFrame with OHLCV data + Date column
            ticker: Stock ticker symbol
        """
        self.df = self._validate_data(price_data)
        self.ticker = ticker
        self._setup_indicators()

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data."""
        required = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        return df.dropna().set_index('Date')

    def _setup_indicators(self):
        """Initialize TA-Lib indicators configuration."""
        self.indicators = {
            'SMA': [20, 50],
            'RSI': 14,
            'MACD': (12, 26, 9),
            'BBANDS': 20,
            'ATR': 14
        }

    def compute_indicators(self) -> None:
        """Calculate all technical indicators."""
        logger.info(f"Computing indicators for {self.ticker}")
        
        closes = self.df['Close'].values
        highs = self.df['High'].values
        lows = self.df['Low'].values
        
        # Moving Averages
        for period in self.indicators['SMA']:
            self.df[f'SMA_{period}'] = ta.SMA(closes, period)
        
        # Oscillators
        self.df['RSI'] = ta.RSI(closes, self.indicators['RSI'])
        macd, signal, _ = ta.MACD(closes, *self.indicators['MACD'])
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = signal
        
        # Volatility
        upper, middle, lower = ta.BBANDS(closes, self.indicators['BBANDS'])
        self.df['BB_Upper'] = upper
        self.df['BB_Lower'] = lower
        self.df['ATR'] = ta.ATR(highs, lows, closes, self.indicators['ATR'])
        
        # Returns
        self.df['Daily_Return'] = self.df['Close'].pct_change()
        self.df['Cumulative_Return'] = (1 + self.df['Daily_Return']).cumprod() - 1

    def get_signals(self) -> Dict[str, pd.Series]:
        """Generate trading signals."""
        return {
            'GoldenCross': self.df['SMA_20'] > self.df['SMA_50'],
            'RSI_Overbought': self.df['RSI'] > 70,
            'RSI_Oversold': self.df['RSI'] < 30,
            'MACD_Bullish': self.df['MACD'] > self.df['MACD_Signal']
        }

    def export_results(self, path: str) -> None:
        """Save analysis results."""
        self.df.to_csv(f"{path}/{self.ticker}_technical.csv")
        logger.info(f"Saved technical analysis for {self.ticker}")