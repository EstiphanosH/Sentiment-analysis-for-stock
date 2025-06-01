import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    def __init__(self, technical_data: Dict[str, pd.DataFrame], 
                 sentiment_data: pd.DataFrame):
        """
        Correlate technical indicators with sentiment.
        
        Args:
            technical_data: Dict of {ticker: technical_df}
            sentiment_data: Processed sentiment DataFrame
        """
        self.technical = technical_data
        self.sentiment = sentiment_data

    def analyze(self) -> Dict[str, pd.DataFrame]:
        """Run correlation analysis for all tickers."""
        results = {}
        for ticker, tech_df in self.technical.items():
            sent_df = self.sentiment[self.sentiment['stock'] == ticker]
            merged = self._merge_data(tech_df, sent_df)
            results[ticker] = self._compute_correlations(merged)
        return results

    def _merge_data(self, tech_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
        """Merge technical and sentiment data."""
        daily_sent = sent_df.resample('D', on='date').agg({
            'sentiment': 'mean',
            'headline': 'count'
        }).rename(columns={'headline': 'news_count'})
        
        return tech_df.join(daily_sent, how='left').dropna()

    def _compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key correlations."""
        return df[['Daily_Return', 'RSI', 'MACD', 'sentiment']].corr()