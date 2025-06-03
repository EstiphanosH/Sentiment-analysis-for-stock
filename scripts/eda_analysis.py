import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
try:
    from ipywidgets import interact, widgets
    from IPython.display import display, clear_output
except ImportError:
    print("⚠️ Warning: ipywidgets or IPython not available. Interactive features will be disabled.")
    # Create dummy functions to prevent errors
    def interact(*args, **kwargs): pass
    def display(*args, **kwargs): pass
    def clear_output(*args, **kwargs): pass
    widgets = type('Widgets', (), {'Dropdown': lambda *args, **kwargs: None})()

from dateutil.parser import parse
import os

class InteractiveDataAnalyzer:
    def __init__(self, df):
        """
        Initialize the analyzer with a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame to analyze
        """
        self.df = df.copy()
        self.numeric_cols = []
        self.datetime_cols = []
        self.categorical_cols = []
        self._initialize_columns()
        self._convert_dtypes()

    def _is_date(self, string):
        try:
            parse(string, fuzzy=False)
            return True
        except:
            return False

    def _initialize_columns(self):
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.datetime_cols = []
        self.categorical_cols = []

        for col in self.df.columns:
            if col in self.numeric_cols:
                continue
            try:
                sample = self.df[col].dropna().astype(str).sample(min(20, len(self.df[col]))).tolist()
                parsed_count = sum([self._is_date(x) for x in sample])
                if parsed_count / len(sample) >= 0.7:
                    self.datetime_cols.append(col)
                else:
                    self.categorical_cols.append(col)
            except:
                self.categorical_cols.append(col)

    def _convert_dtypes(self):
        for col in self.df.columns:
            try:
                if col in self.datetime_cols:
                    self.df[col] = self.df[col].astype(str).apply(
                        lambda x: parse(x, fuzzy=True) if pd.notna(x) else pd.NaT
                    )
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = pd.to_numeric(self.df[col], downcast='float' 
                                                 if np.issubdtype(self.df[col].dtype, np.floating) 
                                                 else 'integer')
            except Exception as e:
                print(f"⚠️ Failed to convert column '{col}' to datetime: {e}")

    def display_head(self, n=5):
        display(self.df.head(n))

    def clean_data(self):
        print("Initial shape:", self.df.shape)
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(axis=0, how='all', inplace=True)
        self.df.dropna(axis=1, how='all', inplace=True)
        print("After cleaning shape:", self.df.shape)

    def show_column_summary(self):
        print("\nNumeric Columns:", self.numeric_cols)
        print("\nDatetime Columns:", self.datetime_cols)
        print("\nCategorical Columns:", self.categorical_cols)

    def describe_numeric(self):
        print("\nDescriptive statistics for numeric columns:")
        display(self.df[self.numeric_cols].describe())

    def describe_categorical(self):
        print("\nTop categories per categorical column:")
        for col in self.categorical_cols:
            print(f"\nColumn: {col}")
            display(self.df[col].value_counts().head(10))

    def missing_values_summary(self):
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        result = pd.DataFrame({'Missing Values': missing, 'Percent (%)': missing_percent})
        result = result[result['Missing Values'] > 0].sort_values(by='Percent (%)', ascending=False)
        print("\nMissing Value Summary:")
        display(result)

    def plot_distribution(self, column):
        if column in self.numeric_cols:
            fig = px.histogram(self.df, x=column, nbins=30, title=f"Distribution of {column}")
        elif column in self.categorical_cols:
            fig = px.bar(self.df[column].value_counts().reset_index(), x='index', y=column,
                         labels={'index': column, column: 'Count'}, title=f"Count of {column}")
        else:
            print(f"Column {column} is not numeric or categorical.")
            return
        fig.show()

    def plot_timeseries(self, time_col, value_col):
        if time_col not in self.datetime_cols or value_col not in self.numeric_cols:
            print("Invalid column types for timeseries plot.")
            return
        fig = px.line(self.df.sort_values(time_col), x=time_col, y=value_col, title=f"{value_col} over Time")
        fig.show()

    def plot_scatter(self, x_col, y_col):
        if x_col in self.numeric_cols and y_col in self.numeric_cols:
            fig = px.scatter(self.df, x=x_col, y=y_col, trendline="ols", title=f"Scatter: {x_col} vs {y_col}")
            fig.show()
        else:
            print("Both x and y columns must be numeric.")

    def plot_boxplots(self):
        for col in self.numeric_cols:
            fig = px.box(self.df, y=col, title=f"Boxplot for {col}")
            fig.show()

    def plot_correlation_matrix(self):
        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns for correlation matrix.")
            return
        corr = self.df[self.numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        fig.show()

    def interactive_distribution_plot(self):
        dropdown = widgets.Dropdown(options=self.numeric_cols + self.categorical_cols,
                                    description='Select Column:')
        interact(self.plot_distribution, column=dropdown)

    def interactive_timeseries_plot(self):
        if not self.datetime_cols or not self.numeric_cols:
            print("No suitable datetime or numeric columns for timeseries plot.")
            return
        time_dropdown = widgets.Dropdown(options=self.datetime_cols, description='Datetime:')
        value_dropdown = widgets.Dropdown(options=self.numeric_cols, description='Value:')
        ui = widgets.HBox([time_dropdown, value_dropdown])

        def update_plot(time_col, value_col):
            self.plot_timeseries(time_col, value_col)

        out = widgets.interactive_output(update_plot, {'time_col': time_dropdown, 'value_col': value_dropdown})
        display(ui, out)

    def interactive_scatter_plot(self):
        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns for scatter plot.")
            return
        x_dropdown = widgets.Dropdown(options=self.numeric_cols, description='X Axis:')
        y_dropdown = widgets.Dropdown(options=self.numeric_cols, description='Y Axis:')
        ui = widgets.HBox([x_dropdown, y_dropdown])

        def update_plot(x_col, y_col):
            self.plot_scatter(x_col, y_col)

        out = widgets.interactive_output(update_plot, {'x_col': x_dropdown, 'y_col': y_dropdown})
        display(ui, out)

    def interactive_summary(self):
        self.show_column_summary()
        self.display_head()
        self.describe_numeric()
        self.describe_categorical()
        self.missing_values_summary()
        self.plot_boxplots()
        self.plot_correlation_matrix()
        self.interactive_distribution_plot()
        self.interactive_timeseries_plot()
        self.interactive_scatter_plot()

if __name__ == '__main__':
    # Example usage
    try:
        # Load sample data
        data_path = '../data/raw/historical/AAPL_historical_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print("✅ Successfully loaded AAPL data")
            
            # Initialize analyzer
            analyzer = InteractiveDataAnalyzer(df)
            
            # Run interactive analysis
            analyzer.interactive_summary()
        else:
            print(f"⚠️ Warning: Sample data file not found at {data_path}")
            print("Please provide a valid DataFrame to use this analyzer.")
            
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        raise  # Re-raise the exception for debugging 