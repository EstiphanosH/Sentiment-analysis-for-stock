import pandas as pd
import numpy as np
import re
import string
import logging
from typing import Union, Optional, Dict
from textblob import TextBlob

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK setup
def download_nltk_data():
    for item in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']:
        try:
            nltk.download(item, quiet=True)
        except Exception as e:
            logger.error(f"Error downloading {item}: {e}")
            raise

download_nltk_data()

# Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

class NewsNLPAnalyzer:
    def __init__(self, data: Union[str, pd.DataFrame], text_column: Optional[str] = None):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Data must be a DataFrame or a CSV path.")

        # Remove unnamed columns
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]
        logger.info(f"Data loaded with columns: {self.data.columns.tolist()}")

        if text_column and text_column not in self.data.columns:
            raise ValueError(f"Column '{text_column}' not found in data.")
        self.text_column = text_column

        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._cache = {}
        self.sentiment_scores = None
        self.topics = None

    def set_text_column(self, column_name: str):
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found.")
        self.text_column = column_name
        self._cache.clear()

    def preprocess_text(self, text: str) -> str:
        if text in self._cache:
            return self._cache[text]

        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@\w+|#', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)

        tokens = tokenizer.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]

        processed = ' '.join(tokens)
        self._cache[text] = processed
        return processed

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        try:
            blob = TextBlob(text)
            vader = self.sia.polarity_scores(text)
            return {
                'textblob_polarity': blob.polarity,
                'textblob_subjectivity': blob.subjectivity,
                'vader_neg': vader['neg'],
                'vader_neu': vader['neu'],
                'vader_pos': vader['pos'],
                'vader_compound': vader['compound'],
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return dict.fromkeys([
                'textblob_polarity', 'textblob_subjectivity',
                'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'
            ], 0.0)

    def calculate_sentiments(self):
        if self.text_column is None:
            raise ValueError("Text column is not set.")

        self.data['processed_text'] = self.data[self.text_column].apply(self.preprocess_text)
        sentiment_df = self.data['processed_text'].apply(lambda x: pd.Series(self.analyze_sentiment(x)))
        self.data = pd.concat([self.data, sentiment_df], axis=1)
        self.sentiment_scores = sentiment_df
        return self.data

    def extract_keywords(self, top_n=20):
        if 'processed_text' not in self.data.columns:
            raise ValueError("Run calculate_sentiments() first.")

        vectorizer = CountVectorizer(max_features=top_n, stop_words='english')
        X = vectorizer.fit_transform(self.data['processed_text'])
        keywords = vectorizer.get_feature_names_out()
        counts = X.sum(axis=0).A1

        self.topics = pd.DataFrame({'keyword': keywords, 'count': counts}).sort_values(by='count', ascending=False)
        return self.topics

    # Interactive Plot: Sentiment Comparison
    def plot_sentiment_comparison(self):
        if self.sentiment_scores is None:
            self.calculate_sentiments()

        df = self.data.melt(
            id_vars=[self.text_column],
            value_vars=['textblob_polarity', 'vader_compound'],
            var_name='Analyzer',
            value_name='Score'
        )

        fig = px.violin(df, x='Analyzer', y='Score', color='Analyzer', box=True, points='all',
                        hover_data=[self.text_column],
                        title='Sentiment Score Comparison')
        fig.show()
        return fig

    # Interactive Plot: VADER Distribution
    def plot_vader_distribution(self):
        if 'vader_compound' not in self.data.columns:
            raise ValueError("Run calculate_sentiments() first.")

        fig = px.histogram(self.data, x='vader_compound', nbins=50,
                           title='VADER Compound Score Distribution',
                           labels={'vader_compound': 'VADER Compound Score'})
        fig.show()
        return fig

    # Interactive Plot: Top Keywords
    def plot_topic_keywords(self):
        if self.topics is None:
            raise ValueError("Run extract_keywords() first.")

        fig = px.bar(self.topics, x='keyword', y='count',
                     title='Top Keywords in Headlines')
        fig.show()
        return fig

    def save_summary_pdf(self, pdf_path="sentiment_summary.pdf"):
        """
        Save a PDF summary of sentiment analysis, including interpretation.
        """
        if self.sentiment_scores is None:
            self.calculate_sentiments()
        if self.topics is None:
            self.extract_keywords()

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # Prepare summary text
        summary = []
        summary.append("Sentiment Analysis Summary Report\n")
        summary.append(f"Total records: {len(self.data)}")
        summary.append(f"Text column: {self.text_column}\n")

        vader_mean = self.data['vader_compound'].mean()
        tb_mean = self.data['textblob_polarity'].mean()
        summary.append(f"Mean VADER Compound: {vader_mean:.3f}")
        summary.append(f"Mean TextBlob Polarity: {tb_mean:.3f}")

        # Interpretation
        if vader_mean > 0.05:
            summary.append("Interpretation: Overall positive sentiment in news.")
        elif vader_mean < -0.05:
            summary.append("Interpretation: Overall negative sentiment in news.")
        else:
            summary.append("Interpretation: Overall neutral sentiment in news.")

        if tb_mean > 0.05:
            summary.append("TextBlob also indicates positive sentiment.")
        elif tb_mean < -0.05:
            summary.append("TextBlob also indicates negative sentiment.")
        else:
            summary.append("TextBlob also indicates neutral sentiment.")

        summary.append("\nTop Keywords:")
        summary.extend([f"{row.keyword}: {row.count}" for row in self.topics.itertuples()])

        # Save to PDF
        with PdfPages(pdf_path) as pdf:
            # Text summary
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0, 1, "\n".join(summary), fontsize=10, va='top', family='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # VADER histogram
            fig = plt.figure(figsize=(8, 4))
            plt.hist(self.data['vader_compound'], bins=50, color='skyblue')
            plt.title('VADER Compound Score Distribution')
            plt.xlabel('VADER Compound')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # TextBlob histogram
            fig = plt.figure(figsize=(8, 4))
            plt.hist(self.data['textblob_polarity'], bins=50, color='orange')
            plt.title('TextBlob Polarity Distribution')
            plt.xlabel('TextBlob Polarity')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Top keywords bar
            fig = plt.figure(figsize=(8, 4))
            plt.bar(self.topics['keyword'], self.topics['count'], color='green')
            plt.title('Top Keywords')
            plt.xlabel('Keyword')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF summary saved to {pdf_path}")