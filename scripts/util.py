import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all price and news data from directory"""
    path = Path(data_dir)
    
    # Load price files (AAPL_prices.csv, etc.)
    price_files = list(path.glob('*_prices.csv'))
    prices = {
        f.stem.split('_')[0].upper(): pd.read_csv(f, parse_dates=['Date'], index_col='Date')
        for f in price_files
    }
    
    # Load consolidated news
    news = pd.read_csv(path / 'consolidated_news.csv', parse_dates=['date'])
    
    return {
        'prices': prices,
        'news': news
    }