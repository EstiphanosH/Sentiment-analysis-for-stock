import numpy as np  # noqa: F401
import pandas as pd
import pandas_ta as ta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Union, Dict, List, Tuple, Any
import logging
from scipy.stats import norm
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use numpy.nan instead of NaN
NaN = np.nan

class TimeSeriesAnalyzer:
    """
    Enhanced time series analysis module with:
    - Technical indicators using pandas_ta
    - Time series decomposition
    - Stationarity testing
    - ARIMA modeling with auto-ARIMA support
    - Interactive visualizations
    - Trend, cycle, and irregular component analysis
    - Complete error handling and logging
    """

    def __init__(self, data: Union[pd.DataFrame, pd.Series], 
                 date_col: Optional[str] = None,
                 value_col: Optional[str] = None,
                 open_col: Optional[str] = None,
                 high_col: Optional[str] = None,
                 low_col: Optional[str] = None,
                 volume_col: Optional[str] = None,
                 freq: Optional[str] = None):
        """
        Initialize the analyzer with validated data.
        
        Args:
            data: Time series data (DataFrame or Series)
            date_col: Column name for dates (if DataFrame)
            value_col: Column name for values (if DataFrame)
            open_col: Column name for open prices
            high_col: Column name for high prices
            low_col: Column name for low prices
            volume_col: Column name for volume data
            freq: Frequency string (e.g., 'D', 'M')
        """
        try:
            self._validate_input(data, date_col, value_col)
            self._prepare_data(data, date_col, value_col, open_col, high_col, low_col, volume_col)
            self._set_frequency(freq)
            
            # Initialize analysis attributes
            self.original_series = self.series.copy()
            self.model = None
            self.model_fit = None
            self.forecast = None
            self.decomposition = None
            self.stationarity_tests = {}
            self.indicators = pd.DataFrame(index=self.ohlcv.index)
            
            logger.info("TimeSeriesAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TimeSeriesAnalyzer: {str(e)}")
            raise

    def _validate_input(self, data: Union[pd.DataFrame, pd.Series],
                        date_col: Optional[str],
                        value_col: Optional[str]) -> None:
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input must be pandas DataFrame or Series")
        if isinstance(data, pd.DataFrame):
            if value_col and value_col not in data.columns:
                raise ValueError(f"Column '{value_col}' not found in DataFrame")
            if date_col and date_col not in data.columns:
                raise ValueError(f"Column '{date_col}' not found in DataFrame")

    def _prepare_data(self, data: Union[pd.DataFrame, pd.Series],
                      date_col: Optional[str],
                      value_col: Optional[str],
                      open_col: Optional[str],
                      high_col: Optional[str],
                      low_col: Optional[str],
                      volume_col: Optional[str]) -> None:
        if isinstance(data, pd.Series):
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Series must have DatetimeIndex")
            self.series = data.copy()
            self.ohlcv = pd.DataFrame({'close': self.series})
        else:
            df = data.copy()
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)

            self.series = df[value_col].copy() if value_col else df.iloc[:, 0].copy()
            self.ohlcv = pd.DataFrame({
                'open': df.get(open_col, np.nan),
                'high': df.get(high_col, np.nan),
                'low': df.get(low_col, np.nan),
                'close': self.series.copy(),
                'volume': df.get(volume_col, np.nan)
            })

    def _set_frequency(self, freq: Optional[str]) -> None:
        if freq:
            self.series = self.series.asfreq(freq)
            self.ohlcv = self.ohlcv.asfreq(freq)
        elif self.series.index.freq is None:
            inferred_freq = pd.infer_freq(self.series.index)
            if inferred_freq:
                self.series = self.series.asfreq(inferred_freq)
                self.ohlcv = self.ohlcv.asfreq(inferred_freq)

    def calculate_technical_indicators(self, **params) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas_ta.
        
        Args:
            **params: Parameters for technical indicators including:
                - sma_period: Simple Moving Average period
                - ema_period: Exponential Moving Average period
                - rsi_period: Relative Strength Index period
                - macd_fast: MACD fast period
                - macd_slow: MACD slow period
                - macd_signal: MACD signal period
                - bollinger_period: Bollinger Bands period
                - bollinger_std: Bollinger Bands standard deviation
        """
        try:
            # Initialize custom strategy
            custom_strategy = ta.Strategy(
                name="Custom Strategy",
                ta=[
                    {"kind": "sma", "length": params.get('sma_period', 20)},
                    {"kind": "ema", "length": params.get('ema_period', 20)},
                    {"kind": "rsi", "length": params.get('rsi_period', 14)},
                    {"kind": "bbands", "length": params.get('bollinger_period', 20), "std": params.get('bollinger_std', 2)},
                    {"kind": "macd", "fast": params.get('macd_fast', 12), "slow": params.get('macd_slow', 26), "signal": params.get('macd_signal', 9)},
                ]
            )
            
            # Calculate indicators
            self.ohlcv.ta.strategy(custom_strategy)
            self.indicators = self.ohlcv.ta.indicators
            
            logger.info("Technical indicators calculated successfully")
            return self.indicators
            
        except Exception as e:
            logger.exception("Failed to calculate technical indicators")
            raise

    def plot_indicators(self, ticker: str = None) -> None:
        """Plot all calculated indicators with interactive subplots."""
        try:
            if self.indicators.empty:
                raise ValueError("No indicators calculated. Run calculate_technical_indicators first.")

            # Create figure with secondary y-axis
            fig = make_subplots(rows=3, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              subplot_titles=('Price and Overlays', 'Momentum', 'Volume'),
                              row_heights=[0.5, 0.25, 0.25])

            # Plot candlesticks
            fig.add_trace(go.Candlestick(
                x=self.ohlcv.index,
                open=self.ohlcv['open'],
                high=self.ohlcv['high'],
                low=self.ohlcv['low'],
                close=self.ohlcv['close'],
                name='OHLC'
            ), row=1, col=1)

            # Add SMA and EMA if available
            if 'SMA_20' in self.indicators.columns:
                fig.add_trace(go.Scatter(
                    x=self.ohlcv.index,
                    y=self.indicators['SMA_20'],
                    name='SMA(20)',
                    line=dict(color='blue')
                ), row=1, col=1)

            if 'EMA_20' in self.indicators.columns:
                fig.add_trace(go.Scatter(
                    x=self.ohlcv.index,
                    y=self.indicators['EMA_20'],
                    name='EMA(20)',
                    line=dict(color='orange')
                ), row=1, col=1)

            # Add Bollinger Bands if available
            if 'BBL_20_2.0' in self.indicators.columns:
                fig.add_trace(go.Scatter(
                    x=self.ohlcv.index,
                    y=self.indicators['BBU_20_2.0'],
                    name='BB Upper',
                    line=dict(color='gray', dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=self.ohlcv.index,
                    y=self.indicators['BBL_20_2.0'],
                    name='BB Lower',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ), row=1, col=1)

            # Add RSI
            if 'RSI_14' in self.indicators.columns:
                fig.add_trace(go.Scatter(
                    x=self.ohlcv.index,
                    y=self.indicators['RSI_14'],
                    name='RSI(14)'
                ), row=2, col=1)
                
                # Add RSI levels
                fig.add_hline(y=70, line_color="red", line_dash="dash", row=2, col=1)
                fig.add_hline(y=30, line_color="green", line_dash="dash", row=2, col=1)

            # Add volume
            fig.add_trace(go.Bar(
                x=self.ohlcv.index,
                y=self.ohlcv['volume'],
                name='Volume'
            ), row=3, col=1)

            # Update layout
            title = f"Technical Analysis - {ticker}" if ticker else "Technical Analysis"
            fig.update_layout(
                title=title,
                yaxis_title="Price",
                yaxis2_title="RSI",
                yaxis3_title="Volume",
                xaxis_rangeslider_visible=False,
                height=900
            )

            fig.show()
            logger.info("Technical indicators plot created successfully")

        except Exception as e:
            logger.exception("Failed to create technical indicators plot")
            raise

    def summarize(self, export_path: Optional[str] = None) -> pd.DataFrame:
        """Generate summary statistics for the time series data."""
        try:
            summary = pd.DataFrame()
            
            # Basic statistics
            basic_stats = self.series.describe()
            for stat, value in basic_stats.items():
                summary.loc['Basic Statistics', stat] = value
            
            # Time-related statistics
            summary.loc['Time Statistics', 'Start Date'] = self.series.index.min()
            summary.loc['Time Statistics', 'End Date'] = self.series.index.max()
            summary.loc['Time Statistics', 'Duration (Days)'] = (self.series.index.max() - self.series.index.min()).days
            summary.loc['Time Statistics', 'Number of Observations'] = len(self.series)
            
            # Missing data statistics
            missing = self.series.isnull().sum()
            summary.loc['Missing Data', 'Missing Values'] = missing
            summary.loc['Missing Data', 'Missing Percentage'] = (missing / len(self.series)) * 100
            
            # Return statistics
            returns = self.series.pct_change().dropna()
            summary.loc['Returns', 'Mean Daily Return'] = returns.mean()
            summary.loc['Returns', 'Daily Return Std'] = returns.std()
            summary.loc['Returns', 'Annualized Volatility'] = returns.std() * np.sqrt(252)
            summary.loc['Returns', 'Annualized Return'] = ((1 + returns.mean()) ** 252) - 1
            
            if export_path:
                summary.to_csv(export_path)
            
            return summary
            
        except Exception as e:
            logger.exception("Failed to generate summary statistics")
            raise

    def plot_time_series(self, title: str = "Time Series Plot") -> None:
        """Plot the time series data with interactive controls."""
        try:
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=self.series,
                mode='lines',
                name='Price'
            ))
            
            # Add range slider and buttons
            fig.update_layout(
                title=title,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            fig.show()
            
        except Exception as e:
            logger.exception("Failed to plot time series")
            raise e

    def plot_decomposition(self, freq: int = 252) -> None:
        """
        Plot time series decomposition using statsmodels.
        
        Args:
            freq: Frequency for seasonal decomposition (default: 252 for daily data)
        """
        try:
            # Perform decomposition
            self.decomposition = seasonal_decompose(
                self.series,
                period=freq,
                extrapolate_trend='freq'
            )

            # Create subplots
            fig = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.05
            )

            # Add traces
            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=self.series,
                name='Original'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=self.decomposition.trend,
                name='Trend'
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=self.decomposition.seasonal,
                name='Seasonal'
            ), row=3, col=1)

            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=self.decomposition.resid,
                name='Residual'
            ), row=4, col=1)

            # Update layout
            fig.update_layout(
                height=900,
                title_text="Time Series Decomposition",
                showlegend=False
            )

            fig.show()
            logger.info("Time series decomposition plot created successfully")

        except Exception as e:
            logger.exception("Failed to create decomposition plot")
            raise

    def test_stationarity(self) -> Dict:
        """
        Perform stationarity tests (ADF and KPSS) on the time series.
        Returns dictionary with test results.
        """
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(self.series.dropna())
            
            # KPSS test
            kpss_result = kpss(self.series.dropna())
            
            self.stationarity_tests = {
                'ADF': {
                    'test_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                },
                'KPSS': {
                    'test_statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05
                }
            }
            
            logger.info("Stationarity tests completed successfully")
            return self.stationarity_tests
            
        except Exception as e:
            logger.exception("Failed to perform stationarity tests")
            raise

    def plot_acf_pacf(self, lags: int = 40) -> None:
        """
        Plot ACF and PACF using matplotlib.
        
        Args:
            lags: Number of lags to plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            plot_acf(self.series, lags=lags, ax=ax1)
            ax1.set_title('Autocorrelation Function')
            
            plot_pacf(self.series, lags=lags, ax=ax2)
            ax2.set_title('Partial Autocorrelation Function')
            
            plt.tight_layout()
            plt.show()
            
            logger.info("ACF and PACF plots created successfully")
            
        except Exception as e:
            logger.exception("Failed to create ACF and PACF plots")
            raise

    def plot_cycle_irregular_components(self, lambda_: int = 1600) -> None:
        """
        Plot cycle and irregular components using HP filter.
        
        Args:
            lambda_: Filter penalty parameter
        """
        try:
            # Apply HP filter
            cycle, trend = hpfilter(self.series, lambda_=lambda_)
            
            # Create figure
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Cyclical Component', 'Trend'),
                              vertical_spacing=0.1)
            
            # Add cycle
            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=cycle,
                name='Cycle'
            ), row=1, col=1)
            
            # Add trend
            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=trend,
                name='Trend'
            ), row=2, col=1)
            
            # Update layout
            fig.update_layout(
                height=600,
                title_text="HP Filter Decomposition",
                showlegend=True
            )
            
            fig.show()
            logger.info("Cycle and irregular components plot created successfully")
            
        except Exception as e:
            logger.exception("Failed to create cycle and irregular components plot")
            raise

    def fit_arima(self, order: Optional[Tuple[int, int, int]] = None,
                  seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                  auto: bool = True) -> Any:
        """
        Fit ARIMA model to the time series.
        
        Args:
            order: ARIMA order (p,d,q) if auto=False
            seasonal_order: Seasonal order (P,D,Q,s) if auto=False
            auto: Whether to use auto_arima for automatic order selection
        """
        try:
            if auto:
                # Use auto_arima to find optimal parameters
                self.model = auto_arima(
                    self.series,
                    seasonal=True,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore"
                )
                logger.info(f"Auto ARIMA selected order: {self.model.order}")
                
            else:
                if order is None:
                    raise ValueError("Must specify order when auto=False")
                    
                self.model = ARIMA(
                    self.series,
                    order=order,
                    seasonal_order=seasonal_order
                )
                self.model = self.model.fit()
                logger.info(f"ARIMA model fitted with order {order}")
            
            return self.model
            
        except Exception as e:
            logger.exception("Failed to fit ARIMA model")
            raise

    def plot_forecast(self, steps: int = 30, alpha: float = 0.05) -> None:
        """
        Plot forecasted values with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals
        """
        try:
            if self.model is None:
                raise ValueError("No ARIMA model fitted. Run fit_arima first.")
                
            # Get forecast
            forecast = self.model.predict(n_periods=steps)
            
            # Create figure
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=self.series.index,
                y=self.series,
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=pd.date_range(start=self.series.index[-1], periods=steps+1)[1:],
                y=forecast,
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title='Time Series Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True
            )
            
            fig.show()
            logger.info("Forecast plot created successfully")
            
        except Exception as e:
            logger.exception("Failed to create forecast plot")
            raise
