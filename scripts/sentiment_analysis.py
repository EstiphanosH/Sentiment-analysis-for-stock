import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, news_data: pd.DataFrame):
        """
        Analyze news sentiment and topics.
        
        Args:
            news_data: DataFrame with 'date', 'headline', 'publisher', 'stock'
        """
        self.df = self._preprocess(news_data)
        self.stopwords = set(pd.read_csv('data/stopwords.csv')['word'])  # Custom financial stopwords

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare news data."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['tokens'] = df['headline'].apply(self._tokenize)
        return df.dropna()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        text = re.sub(r'[^\w\s]|[\d]', '', str(text).lower())
        return [word for word in text.split() if word not in self.stopwords]

    def analyze_sentiment(self) -> None:
        """Calculate sentiment scores (placeholder for actual sentiment model)."""
        logger.info("Calculating sentiment scores")
        # Replace with actual sentiment analysis (VADER, etc.)
        self.df['sentiment'] = np.random.uniform(-1, 1, len(self.df))  # Mock data

    def analyze_topics(self, n_topics: int = 5) -> Dict[int, List[str]]:
        """Perform topic modeling."""
        logger.info(f"Identifying {n_topics} topics")
        
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(self.df['tokens'].apply(' '.join))
        
        model = NMF(n_components=n_topics, random_state=42)
        model.fit(X)
        
        topics = {}
        for i, topic in enumerate(model.components_):
            topics[i+1] = [vectorizer.get_feature_names_out()[j] 
                          for j in topic.argsort()[-5:][::-1]]
        
        self.df['dominant_topic'] = model.transform(X).argmax(axis=1) + 1
        return topics

    def get_publisher_stats(self) -> pd.DataFrame:
        """Analyze publishers by sentiment and volume."""
        return self.df.groupby('publisher').agg({
            'headline': 'count',
            'sentiment': ['mean', 'std']
        }).sort_values(('headline', 'count'), ascending=False)