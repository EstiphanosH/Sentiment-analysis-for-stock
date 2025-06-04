import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import plotly.express as px

class CorrelationAnalyzer:
    """
    Correlation analysis between price (OCHL), analyst rating data, and sentiment data.
    """

    def __init__(self, data_dir="data/raw", analyst_file="raw_analyst_ratings.csv"):
        self.data_dir = data_dir
        self.analyst_file = analyst_file
        self.price_data = None
        self.analyst_data = None
        self.sentiment_data = None
        self.merged_data = None
        self.correlation_results = {}

    def load_price_data(self, ticker: str, date_col="Date"):
        """
        Load historical price data for a given ticker.
        """
        path = os.path.join(self.data_dir, "historical", f"{ticker}_historical_data.csv")
        self.price_data = pd.read_csv(path, parse_dates=[date_col])
        self.price_data = self.price_data.sort_values(date_col)
        self.price_data = self.price_data.set_index(date_col)

    def load_analyst_data(self, date_col="date"):
        """
        Load analyst rating data.
        """
        path = os.path.join(self.data_dir, self.analyst_file)
        self.analyst_data = pd.read_csv(path, parse_dates=[date_col])
        self.analyst_data = self.analyst_data.sort_values(date_col)
        self.analyst_data = self.analyst_data.set_index(date_col)

    def load_sentiment_data(self, sentiment_df, date_col="date", sentiment_col="sentiment"):
        """
        Load sentiment data (already aggregated by date).
        """
        self.sentiment_data = sentiment_df[[date_col, sentiment_col]].copy()
        self.sentiment_data = self.sentiment_data.sort_values(date_col)
        self.sentiment_data = self.sentiment_data.set_index(date_col)

    def merge_data(self, how="inner"):
        """
        Merge price and analyst data on date.
        """
        if self.price_data is None or self.analyst_data is None:
            raise ValueError("Both price and analyst data must be loaded before merging.")
        self.merged_data = pd.merge(
            self.price_data, self.analyst_data, left_index=True, right_index=True, how=how
        )

    def merge_with_sentiment(self, how="inner"):
        """
        Merge price data with sentiment data on date.
        """
        if self.price_data is None or self.sentiment_data is None:
            raise ValueError("Both price and sentiment data must be loaded before merging.")
        self.merged_data = pd.merge(
            self.price_data, self.sentiment_data, left_index=True, right_index=True, how=how
        )

    def _is_valid_numeric_column(self, col):
        return (
            col in self.merged_data.columns and
            pd.api.types.is_numeric_dtype(self.merged_data[col])
        )

    def analyze_correlations(self, ochl_columns: list, analyst_columns: list, method='pearson') -> dict:
        """
        Correlation analysis between OCHL and analyst/sentiment columns.
        Supported methods: 'pearson' (default), 'spearman'.
        """
        if self.merged_data is None:
            raise ValueError("Merged data must be available for correlation analysis.")

        self.correlation_results.clear()

        if method == 'pearson':
            corr_func = pearsonr
        elif method == 'spearman':
            corr_func = spearmanr
        else:
            raise ValueError(f"Unsupported correlation method: {method}")

        for o_col in ochl_columns:
            if not self._is_valid_numeric_column(o_col):
                continue
            self.correlation_results[o_col] = {}
            for a_col in analyst_columns:
                if not self._is_valid_numeric_column(a_col):
                    continue
                valid_data = self.merged_data[[o_col, a_col]].dropna()
                if len(valid_data) < 2:
                    continue
                corr, p_val = corr_func(valid_data[o_col], valid_data[a_col])
                self.correlation_results[o_col][a_col] = {
                    "correlation": corr,
                    "p_value": p_val,
                    "n": len(valid_data)
                }
        return self.correlation_results

    def plot_correlation_matrix(self, columns: list):
        """
        Interactive heatmap for correlations among selected columns in merged data.
        """
        if self.merged_data is None:
            raise ValueError("Merged data must be available for plotting.")

        numeric_cols = [col for col in columns if self._is_valid_numeric_column(col)]
        if not numeric_cols:
            raise ValueError("No valid numeric columns found for correlation matrix.")

        corr_matrix = self.merged_data[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            text_auto=True
        )
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables",
            coloraxis_colorbar=dict(title="Correlation")
        )
        return fig

    def plot_scatter_matrix(self, columns: list):
        """
        Interactive scatter matrix for selected columns in merged data.
        """
        if self.merged_data is None:
            raise ValueError("Merged data must be available for plotting.")

        numeric_cols = [col for col in columns if self._is_valid_numeric_column(col)]
        if not numeric_cols:
            raise ValueError("No valid numeric columns found for scatter matrix.")

        fig = px.scatter_matrix(
            self.merged_data[numeric_cols],
            title="Scatter Matrix"
        )
        fig.update_layout(hovermode="closest")
        return fig