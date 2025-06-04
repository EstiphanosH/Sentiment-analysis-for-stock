import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Try importing interactive widgets and display
try:
    from ipywidgets import interact, widgets
    from IPython.display import display, clear_output
    _interactive_available = True
except ImportError:
    print("ipywidgets or IPython not available. Interactive features will be disabled.")
    _interactive_available = False
    def interact(*args, **kwargs): pass
    def display(obj): print(obj)
    def clear_output(*args, **kwargs): pass
    class DummyWidgets:
        def Dropdown(self, *args, **kwargs): return None
        def HBox(self, *args, **kwargs): return None
        def interactive_output(self, *args, **kwargs): return None
    widgets = DummyWidgets()

class InteractiveDataAnalyzer:
    """
    General EDA for tabular (stock) and news/text data.
    """
    def __init__(self, df, save_dir=None):
        self.df = df.copy()
        self.save_dir = save_dir or "eda_reports"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include='object').columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include='datetime').columns.tolist()
        # Try to parse any date columns
        for col in self.df.columns:
            if 'date' in col.lower() and col not in self.datetime_cols:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.datetime_cols.append(col)
                except Exception:
                    pass

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
        if self.numeric_cols:
            display(self.df[self.numeric_cols].describe())
        else:
            print("No numeric columns found.")

    def describe_categorical(self):
        print("\nTop categories per categorical column:")
        if not self.categorical_cols:
            print("No categorical columns found.")
        for col in self.categorical_cols:
            print(f"\nColumn: {col}")
            display(self.df[col].value_counts().head(10))

    def missing_values_summary(self):
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        result = pd.DataFrame({'Missing Values': missing, 'Percent (%)': missing_percent})
        result = result[result['Missing Values'] > 0].sort_values(by='Percent (%)', ascending=False)
        print("\nMissing Value Summary:")
        if not result.empty:
            display(result)
        else:
            print("No missing values.")

    def plot_distribution(self, column, save=False):
        fig = None
        if column in self.numeric_cols:
            fig = px.histogram(self.df, x=column, nbins=30, title=f"Distribution of {column}")
        elif column in self.categorical_cols:
            fig = px.bar(self.df[column].value_counts().reset_index(), x='index', y=column,
                         labels={'index': column, column: 'Count'}, title=f"Count of {column}")
        else:
            print(f"Column {column} is not numeric or categorical.")
            return
        fig.show()
        if save and self.save_dir:
            out_path = os.path.join(self.save_dir, f"distribution_{column}.png")
            fig.write_image(out_path)
            print(f"Saved: {out_path}")

    def plot_timeseries(self, time_col, value_col, save=False):
        if time_col not in self.datetime_cols or value_col not in self.numeric_cols:
            print("Invalid column types for timeseries plot.")
            return
        fig = px.line(self.df.sort_values(time_col), x=time_col, y=value_col, title=f"{value_col} over Time")
        fig.show()
        if save and self.save_dir:
            out_path = os.path.join(self.save_dir, f"timeseries_{time_col}_{value_col}.png")
            fig.write_image(out_path)
            print(f"Saved: {out_path}")

    def plot_scatter(self, x_col, y_col, save=False):
        if x_col == y_col:
            print("❌ X and Y columns must be different for scatter plot.")
            return
        if x_col in self.numeric_cols and y_col in self.numeric_cols:
            fig = px.scatter(self.df, x=x_col, y=y_col, trendline="ols", title=f"Scatter: {x_col} vs {y_col}")
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, f"scatter_{x_col}_{y_col}.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")
        else:
            print("Both x and y columns must be numeric.")

    def plot_boxplots(self, save=False):
        for col in self.numeric_cols:
            fig = px.box(self.df, y=col, title=f"Boxplot for {col}")
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, f"boxplot_{col}.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")

    def plot_correlation_matrix(self, save=False):
        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns for correlation matrix.")
            return
        corr = self.df[self.numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        fig.show()
        if save and self.save_dir:
            out_path = os.path.join(self.save_dir, "correlation_matrix.png")
            fig.write_image(out_path)
            print(f"Saved: {out_path}")

    def plot_headline_length(self, save=False):
        if 'headline' in self.df.columns:
            self.df['headline_length'] = self.df['headline'].astype(str).apply(len)
            fig = px.histogram(self.df, x='headline_length', nbins=30, title='Headline Length Distribution')
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, "headline_length_dist.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")

    def plot_publishers(self, save=False):
        if 'publisher' in self.df.columns:
            fig = px.bar(self.df['publisher'].value_counts().head(20).reset_index(),
                         x='index', y='publisher', title='Top 20 Publishers',
                         labels={'index': 'Publisher', 'publisher': 'Article Count'})
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, "top_publishers.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")

    def plot_publication_dates(self, save=False):
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['date_only'] = self.df['date'].dt.date
            pub_counts = self.df.groupby('date_only').size()
            fig = px.line(pub_counts, title='Articles Published per Day')
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, "articles_per_day.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")

    def plot_publish_times(self, save=False):
        if 'date' in self.df.columns:
            self.df['hour'] = pd.to_datetime(self.df['date'], errors='coerce').dt.hour
            fig = px.histogram(self.df, x='hour', nbins=24, title='Articles by Hour of Day')
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, "articles_by_hour.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")

    def plot_publisher_domains(self, save=False):
        if 'publisher' in self.df.columns and self.df['publisher'].str.contains('@').any():
            self.df['domain'] = self.df['publisher'].str.extract(r'@([\w\.-]+)')
            fig = px.bar(self.df['domain'].value_counts().head(10).reset_index(),
                         x='index', y='domain', title='Top 10 Publisher Domains',
                         labels={'index': 'Domain', 'domain': 'Article Count'})
            fig.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, "top_publisher_domains.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")

    def plot_headline_keywords(self, save=False):
        if 'headline' in self.df.columns:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(stop_words='english', max_features=20)
            X = vectorizer.fit_transform(self.df['headline'].astype(str))
            keywords = vectorizer.get_feature_names_out()
            freqs = X.sum(axis=0).A1
            keyword_freq = dict(zip(keywords, freqs))
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(keyword_freq.values()), y=list(keyword_freq.keys()))
            plt.title('Top 20 Keywords in Headlines')
            plt.xlabel('Frequency')
            plt.tight_layout()
            plt.show()
            if save and self.save_dir:
                out_path = os.path.join(self.save_dir, "headline_keywords.png")
                plt.savefig(out_path)
                print(f"Saved: {out_path}")
                plt.close()

    def interactive_distribution_plot(self):
        if not _interactive_available:
            print("Interactive widgets not available.")
            return
        dropdown = widgets.Dropdown(options=self.numeric_cols + self.categorical_cols,
                                   description='Select Column:')
        def plot_and_save(column):
            self.plot_distribution(column, save=True)
        interact(plot_and_save, column=dropdown)

    def interactive_timeseries_plot(self):
        if not _interactive_available:
            print("Interactive widgets not available.")
            return
        if not self.datetime_cols or not self.numeric_cols:
            print("No suitable datetime or numeric columns for timeseries plot.")
            return
        # Fix the first datetime column as the time axis
        time_col = self.datetime_cols[0]
        value_dropdown = widgets.Dropdown(options=self.numeric_cols, description='Value:')
        ui = widgets.HBox([widgets.Label(f"Time: {time_col}"), value_dropdown])

        def update_plot(value_col):
            self.plot_timeseries(time_col, value_col, save=True)

        value_dropdown.observe(lambda change: update_plot(value_dropdown.value), names='value')

        display(ui)
        # Initial plot
        update_plot(value_dropdown.value)

    def interactive_scatter_plot(self):
        if not _interactive_available:
            print("Interactive widgets not available.")
            return
        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns for scatter plot.")
            return
        x_dropdown = widgets.Dropdown(options=self.numeric_cols, description='X Axis:')
        y_dropdown = widgets.Dropdown(options=self.numeric_cols, description='Y Axis:')
        ui = widgets.HBox([x_dropdown, y_dropdown])

        def update_plot(*args):
            x_col = x_dropdown.value
            y_col = y_dropdown.value
            if x_col == y_col:
                print("❌ X and Y columns must be different for scatter plot.")
                return
            self.plot_scatter(x_col, y_col, save=True)

        x_dropdown.observe(update_plot, names='value')
        y_dropdown.observe(update_plot, names='value')

        display(ui)
        update_plot()

    def interactive_summary(self, save_pdf: bool = False, pdf_path: str = None):
        """
        Display interactive summary. Optionally save summary as a PDF file.
        """
        import io
        import matplotlib.pyplot as plt

        print("DataFrame shape:", self.df.shape)
        print("\nFirst 5 rows:")
        display(self.df.head())
        print("\nDataFrame info:")
        buf = io.StringIO()
        self.df.info(buf=buf)
        info_str = buf.getvalue()
        print(info_str)
        self.show_column_summary()
        self.describe_numeric()
        self.describe_categorical()
        self.missing_values_summary()
        self.plot_boxplots(save=True)
        self.plot_correlation_matrix(save=True)
        self.interactive_distribution_plot()
        self.interactive_timeseries_plot()
        self.interactive_scatter_plot()

        # News-specific EDA if columns exist
        self.plot_headline_length(save=True)
        self.plot_publishers(save=True)
        self.plot_publication_dates(save=True)
        self.plot_publish_times(save=True)
        self.plot_publisher_domains(save=True)
        self.plot_headline_keywords(save=True)

        if save_pdf:
            from matplotlib.backends.backend_pdf import PdfPages
            if pdf_path is None:
                pdf_path = "../reports/pdfs/eda_summary.pdf"

            # Save textual summary to a temporary file
            text_summary = io.StringIO()
            text_summary.write(f"DataFrame shape: {self.df.shape}\n\n")
            text_summary.write("First 5 rows:\n")
            text_summary.write(self.df.head().to_string())
            text_summary.write("\n\nDataFrame info:\n")
            text_summary.write(info_str)
            text_summary.write("\n\nNumeric Columns:\n")
            text_summary.write(str(self.numeric_cols))
            text_summary.write("\n\nDatetime Columns:\n")
            text_summary.write(str(self.datetime_cols))
            text_summary.write("\n\nCategorical Columns:\n")
            text_summary.write(str(self.categorical_cols))
            text_summary.write("\n\nDescriptive statistics for numeric columns:\n")
            if self.numeric_cols:
                text_summary.write(str(self.df[self.numeric_cols].describe()))
            else:
                text_summary.write("No numeric columns found.")
            text_summary.write("\n\nTop categories per categorical column:\n")
            if not self.categorical_cols:
                text_summary.write("No categorical columns found.")
            for col in self.categorical_cols:
                text_summary.write(f"\nColumn: {col}\n")
                text_summary.write(str(self.df[col].value_counts().head(10)))
            text_summary.write("\n\nMissing Value Summary:\n")
            missing = self.df.isnull().sum()
            missing_percent = (missing / len(self.df)) * 100
            result = pd.DataFrame({'Missing Values': missing, 'Percent (%)': missing_percent})
            result = result[result['Missing Values'] > 0].sort_values(by='Percent (%)', ascending=False)
            if not result.empty:
                text_summary.write(str(result))
            else:
                text_summary.write("No missing values.")

            # Create a PDF and add the text summary as a page
            with PdfPages(pdf_path) as pdf:
                # Text summary as a figure
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                txt = text_summary.getvalue()
                ax.text(0, 1, txt, fontsize=8, va='top', family='monospace')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # Save all PNG plots in save_dir to the PDF
                if self.save_dir and os.path.exists(self.save_dir):
                    import glob
                    for img_path in glob.glob(os.path.join(self.save_dir, "*.png")):
                        img = plt.imread(img_path)
                        fig, ax = plt.subplots(figsize=(8.5, 6))
                        ax.imshow(img)
                        ax.axis('off')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            print(f"PDF summary saved to {pdf_path}")

# Usage in notebook:
# from scripts.eda_analysis import InteractiveDataAnalyzer
# analyzer = InteractiveDataAnalyzer(dataframes['AAPL_historical_data'])
# analyzer.interactive_summary(save_pdf=True, pdf_path="eda_aapl.pdf")
# analyzer = InteractiveDataAnalyzer(dataframes['raw_analyst_ratings'])
# analyzer.interactive_summary(save_pdf=True, pdf_path="eda_news.pdf")