#!/usr/bin/env python3
# main.py - Optimized financial analysis pipeline

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# Local imports
from scripts.quantitative_analysis import QuantitativeAnalysis
from scripts.sentiment_analysis import FinancialSentimentAnalyzer
from scripts.correlation_analysis import CorrelationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(ticker: str, data_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """
    Load required datasets for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        data_dir: Directory containing data files
    
    Returns:
        Dictionary with 'price' and 'news' DataFrames
    """
    data_dir = Path(data_dir)
    try:
        # Load price data
        price_path = data_dir / f'{ticker.lower()}_prices.csv'
        price_df = pd.read_csv(
            price_path,
            parse_dates=['Date'],
            usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        )
        
        # Load news data
        news_path = data_dir / f'{ticker.lower()}_news.csv'
        news_df = pd.read_csv(
            news_path,
            parse_dates=['date'],
            usecols=['date', 'headline', 'publisher', 'sentiment', 'stock']
        )
        
        return {
            'price': price_df,
            'news': news_df
        }
    except Exception as e:
        logger.error(f"Failed to load data for {ticker}: {str(e)}")
        raise

def run_quantitative_analysis(price_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """Run quantitative technical analysis."""
    logger.info(f"Running quantitative analysis for {ticker}")
    qa = QuantitativeAnalysis(price_df, ticker)
    qa.calculate_indicators()
    
    # Generate and save reports
    report = qa.generate_report()
    qa.export_results(f'results/{ticker}_quantitative')
    
    return report

def run_sentiment_analysis(news_df: pd.DataFrame, price_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """Run news sentiment analysis."""
    logger.info(f"Running sentiment analysis for {ticker}")
    sa = FinancialSentimentAnalyzer(news_df, price_df)
    sa.analyze_news_sentiment()
    
    # Generate correlation insights
    merged = sa.merge_with_stock_data()
    merged.to_csv(f'results/{ticker}_sentiment_correlation.csv', index=False)
    
    return {
        'avg_sentiment': merged['DailySentiment'].mean(),
        'correlation': merged[['Daily_Return', 'DailySentiment']].corr().iloc[0,1]
    }

def run_full_analysis(ticker: str) -> Dict[str, Any]:
    """
    Execute complete analysis pipeline for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing all analysis results
    """
    try:
        # Load data
        data = load_data(ticker)
        price_df, news_df = data['price'], data['news']
        
        # Run analyses
        quant_report = run_quantitative_analysis(price_df, ticker)
        sentiment_report = run_sentiment_analysis(news_df, price_df, ticker)
        
        # Combine results
        return {
            'ticker': ticker,
            'quantitative': quant_report,
            'sentiment': sentiment_report,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main execution function."""
    # Create output directory
    Path('results').mkdir(exist_ok=True)
    
    # Analyze multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
    results = []
    
    for ticker in tickers:
        logger.info(f"Starting analysis for {ticker}")
        result = run_full_analysis(ticker)
        results.append(result)
        
        if result['status'] == 'success':
            logger.info(f"Completed analysis for {ticker}")
        else:
            logger.warning(f"Failed analysis for {ticker}")
    
    # Generate summary report
    summary = pd.DataFrame([
        {
            'Ticker': r['ticker'],
            'Total Return (%)': r.get('quantitative', {}).get('performance', {}).get('total_return', 0) * 100,
            'Sentiment Correlation': r.get('sentiment', {}).get('correlation', 0),
            'Status': r['status']
        }
        for r in results
    ])
    
    summary.to_csv('results/summary_report.csv', index=False)
    logger.info("Analysis complete. Summary report saved to results/summary_report.csv")

if __name__ == '__main__':
    main()