import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Suppress interpolation and statsmodels warnings globally
warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message="No supported index is available. Prediction results will be given with an integer index beginning at `start`.")
warnings.filterwarnings("ignore", message="No supported index is available. In the next version, calling this method in a model without a supported")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
class TimeSeriesAnalyzer:
    def __init__(self, data, date_col, value_col, open_col=None, high_col=None, low_col=None, volume_col=None):
        self.df = data.copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        self.df = self.df.sort_values(date_col)
        self.df.set_index(date_col, inplace=True)
        # Try to infer and set frequency for the index
        freq = pd.infer_freq(self.df.index)
        if freq:
            self.df = self.df.asfreq(freq)
        else:
            # If frequency can't be inferred, use business day as default
            self.df = self.df.asfreq('B')
        self.series = self.df[value_col]
        self.value_col = value_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col

    def plot_series(self, title="Time Series Plot"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.series.index, y=self.series, mode='lines', name=self.value_col))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=self.value_col,
            xaxis=dict(rangeslider=dict(visible=True))
        )
        fig.show()

    def plot_decomposition(self, freq=252):
        # Drop missing values for decomposition
        clean_series = self.series.dropna()
        result = seasonal_decompose(clean_series, period=freq, extrapolate_trend='freq')
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))
        fig.add_trace(go.Scatter(x=clean_series.index, y=result.observed, name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=clean_series.index, y=result.trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=clean_series.index, y=result.seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=clean_series.index, y=result.resid, name='Residual'), row=4, col=1)
        fig.update_layout(height=900, title_text="Time Series Decomposition", showlegend=False)
        fig.show()
        return result

    def test_stationarity(self):
        adf = adfuller(self.series.dropna())
        kpss_stat, kpss_p, *_ = kpss(self.series.dropna(), nlags="auto")
        return {
            "ADF": {"statistic": float(adf[0]), "p-value": float(adf[1])},
            "KPSS": {"statistic": float(kpss_stat), "p-value": float(kpss_p)}
        }

    def plot_acf_pacf(self, lags=40):
        acf_vals = acf(self.series.dropna(), nlags=lags)
        pacf_vals = pacf(self.series.dropna(), nlags=lags)
        fig = make_subplots(rows=2, cols=1, subplot_titles=('ACF', 'PACF'))
        fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF'), row=2, col=1)
        fig.update_layout(height=600, title_text="ACF and PACF")
        fig.show()

    def calculate_technical_indicators(self, sma_period=20, ema_period=20, rsi_period=14,
                                       macd_fast=12, macd_slow=26, macd_signal=9,
                                       bollinger_period=20, bollinger_std=2):
        df = self.df.copy()
        # SMA & EMA
        df['SMA'] = df[self.value_col].rolling(window=sma_period).mean()
        df['EMA'] = df[self.value_col].ewm(span=ema_period, adjust=False).mean()
        # RSI
        delta = df[self.value_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # MACD
        ema_fast = df[self.value_col].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df[self.value_col].ewm(span=macd_slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
        # Bollinger Bands
        rolling_mean = df[self.value_col].rolling(window=bollinger_period).mean()
        rolling_std = df[self.value_col].rolling(window=bollinger_period).std()
        df['Bollinger_Upper'] = rolling_mean + (rolling_std * bollinger_std)
        df['Bollinger_Lower'] = rolling_mean - (rolling_std * bollinger_std)
        return df

    def fit_arima(self, order=(1,1,1)):
        model = ARIMA(self.series, order=order)
        self.model_fit = model.fit()
        print(self.model_fit.summary())
        return self.model_fit

    def forecast(self, steps=30):
        forecast = self.model_fit.get_forecast(steps=steps)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        # Use the index with frequency for future dates
        last_date = self.series.index[-1]
        freq = self.series.index.freq or pd.infer_freq(self.series.index)
        if freq is None:
            freq = 'B'
        future_index = pd.date_range(last_date, periods=steps+1, freq=freq)[1:]
        pred.index = future_index
        conf_int.index = future_index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.series.index, y=self.series, mode='lines', name='Observed'))
        fig.add_trace(go.Scatter(x=future_index, y=pred, mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(
            x=future_index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line=dict(color='lightgrey'), name='Lower CI'))
        fig.add_trace(go.Scatter(
            x=future_index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', line=dict(color='lightgrey'), name='Upper CI'))
        fig.update_layout(title="Forecast", xaxis_title="Date", yaxis_title=self.value_col)
        fig.show()
        return pred

    def save_summary_pdf(self, pdf_path="tsa_summary.pdf"):
        import io
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        buf = io.StringIO()
        buf.write(f"Time Series Summary Report\n\n")
        buf.write(f"Series length: {len(self.series)}\n")
        buf.write(f"Start date: {self.series.index.min()}\n")
        buf.write(f"End date: {self.series.index.max()}\n\n")
        stat = self.test_stationarity()
        buf.write(f"ADF Statistic: {stat['ADF']['statistic']:.4f}, p-value: {stat['ADF']['p-value']:.4g}\n")
        if stat['ADF']['p-value'] > 0.05:
            buf.write("Interpretation: The series is likely non-stationary (fail to reject H0).\n")
        else:
            buf.write("Interpretation: The series is likely stationary (reject H0).\n")
        buf.write(f"KPSS Statistic: {stat['KPSS']['statistic']:.4f}, p-value: {stat['KPSS']['p-value']:.4g}\n")
        if stat['KPSS']['p-value'] < 0.05:
            buf.write("Interpretation: The series is likely non-stationary (reject H0).\n")
        else:
            buf.write("Interpretation: The series is likely stationary (fail to reject H0).\n")
        buf.write("\nLast 5 values:\n")
        buf.write(self.series.tail().to_string())
        buf.write("\n")

        with PdfPages(pdf_path) as pdf:
            # Text summary
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0, 1, buf.getvalue(), fontsize=10, va='top', family='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Time series plot
            fig = plt.figure(figsize=(10, 4))
            plt.plot(self.series)
            plt.title("Time Series")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Decomposition
            # In save_summary_pdf
            result = seasonal_decompose(self.series.dropna(), period=252, extrapolate_trend='freq')
            result.plot()
            plt.suptitle("Decomposition", y=1.02)
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()

            # ACF/PACF
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            plot_acf(self.series.dropna(), lags=40, ax=axes[0])
            plot_pacf(self.series.dropna(), lags=40, ax=axes[1])
            axes[0].set_title("ACF")
            axes[1].set_title("PACF")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF summary saved to {pdf_path}")