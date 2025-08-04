import pandas as pd
import numpy as np
import talib
from pandas.tseries.offsets import BDay

class FeatureEngineer:
    """Generates the features for the model."""
    def __init__(self):
        """
        Initializes new variables to store the data
        """
        self.ticker = None
        self.ohlcv_df = None
        self.fundamentals_df = None
        self.info_df = None
        self.ratings_df = None
    
    def load_data(self, ticker: str) -> None:
        """Load all the collected data from the excel and parquet files into dataframes

        Args:
            ticker (str): Ticker symbol
        """
        self.ticker = ticker
        self.ohlcv_df = pd.read_excel(f'historical_data/xlsx/{ticker}.xlsx')
        self.fundamentals_df = pd.read_excel(f'fundamentals/{ticker}/fundamentals.xlsx')
        self.info_df = pd.read_excel(f'fundamentals/{ticker}/info.xlsx')
        self.ratings_df = pd.read_excel(f'fundamentals/{ticker}/ratings.xlsx')
    
    def calculate_fundamental_features_timeseries(self, ticker: str) -> pd.DataFrame:
        """Calculate all fundamental features from parsed data
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            Dataframe: Returns the fundamental features in a dataframe. Fundamentals, info and analyst ratings are all appended in the same df.
        """
        # Get all date columns (excluding 'Source' and 'index')
        date_cols = [col for col in self.fundamentals_df.columns if col not in ['Source', 'index']]

        # Helper to get a Series for each fundamental (indexed by date)
        def get_fund_value(row_name):
            row = self.fundamentals_df.loc[self.fundamentals_df['index'] == row_name, date_cols]
            if row.empty:
                return pd.Series([np.nan] * len(date_cols), index=date_cols)
            return row.squeeze().astype(np.float64)
        
        features = {
            'return_on_assets': get_fund_value('NetIncome') / get_fund_value('TotalAssets'),
            'return_on_equity': get_fund_value('NetIncome') / get_fund_value('StockholdersEquity'),
            'cash_conversion': get_fund_value('OperatingCashFlow') / get_fund_value('NetIncome'),
            'fcf_margin': get_fund_value('FreeCashFlow') / get_fund_value('TotalRevenue'),
            'ocf_margin': get_fund_value('OperatingCashFlow') / get_fund_value('TotalRevenue'),
            'fcf_conversion': get_fund_value('FreeCashFlow') / get_fund_value('NetIncome'),
            'capex_intensity': abs(get_fund_value('CapitalExpenditure')) / get_fund_value('TotalRevenue'),
            'reinvestment_rate': abs(get_fund_value('CapitalExpenditure')) / get_fund_value('OperatingCashFlow'),
            'fcf_after_capex': get_fund_value('OperatingCashFlow') - abs(get_fund_value('CapitalExpenditure')),
            'current_ratio': get_fund_value('CurrentAssets') / get_fund_value('CurrentLiabilities'),
            'cash_ratio': get_fund_value('CashAndCashEquivalents') / get_fund_value('CurrentLiabilities'),
            'debt_to_equity': get_fund_value('TotalDebt') / get_fund_value('StockholdersEquity'),
            'debt_to_assets': get_fund_value('TotalDebt') / get_fund_value('TotalAssets'),
            'equity_ratio': get_fund_value('StockholdersEquity') / get_fund_value('TotalAssets'),
            'tangible_equity_ratio': get_fund_value('TangibleBookValue') / get_fund_value('TotalAssets'),
            'asset_turnover': get_fund_value('TotalRevenue') / get_fund_value('TotalAssets'),
            'receivables_turnover': get_fund_value('TotalRevenue') / get_fund_value('AccountsReceivable'),
            'working_capital_ratio': (get_fund_value('CurrentAssets') - get_fund_value('CurrentLiabilities')) / get_fund_value('TotalAssets'),
            'book_value_per_share': get_fund_value('StockholdersEquity') / get_fund_value('OrdinarySharesNumber'),
            'tangible_book_per_share': get_fund_value('TangibleBookValue') / get_fund_value('OrdinarySharesNumber'),
            'revenue_per_share': get_fund_value('TotalRevenue') / get_fund_value('OrdinarySharesNumber'),
            'operating_margins': get_fund_value('OperatingIncome') / get_fund_value('TotalRevenue'),
            'ebitda_margins': get_fund_value('EBITDA') / get_fund_value('TotalRevenue'),
            'gross_margins': get_fund_value('GrossProfit') / get_fund_value('TotalRevenue'),
            'net_margins': get_fund_value('NetIncome') / get_fund_value('TotalRevenue')
        }
        
        # Combine all features into a DataFrame (features as rows, dates as columns)
        features_df = pd.DataFrame(features).T
        features_df.columns = pd.to_datetime(features_df.columns)
        features_df.index.name = "Feature"
        features_df = features_df.iloc[:, :-1]

        # Save to Excel
        features_df.to_excel('testing_data/fundamentals_features_timeseries.xlsx')
        
        return features_df
    
    def create_features(self, ticker: str) -> pd.DataFrame:
        """Compiles all the data and creates the features

        Args:
            ticker (str): The stock ticker

        Returns:
            pd.DataFrame: Dataframe with compiled features (needs more processing before feeding it to a model)
        """
        self.load_data(ticker)
        
        self.ohlcv_df['date'] = pd.to_datetime(self.ohlcv_df['date'])
        self.ohlcv_df.set_index('date', inplace=True)
        
        # Extract OHLCV arrays
        close = self.ohlcv_df['close'].values.astype(np.float64)
        high = self.ohlcv_df['high'].values.astype(np.float64)
        low = self.ohlcv_df['low'].values.astype(np.float64)
        open_price = self.ohlcv_df['open'].values.astype(np.float64)
        volume = self.ohlcv_df['volume'].values.astype(np.float64)
            
        # Create series
        close_series = pd.Series(close, index=self.ohlcv_df.index)
        volume_series = pd.Series(volume, index=self.ohlcv_df.index)
        high_series = pd.Series(high, index=self.ohlcv_df.index)
        low_series = pd.Series(low, index=self.ohlcv_df.index)
        open_series = pd.Series(open_price, index=self.ohlcv_df.index)
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=self.ohlcv_df.index)

        # Raw OHLCV data
        features['Open'] = open_series
        features['High'] = high_series  
        features['Low'] = low_series
        features['Close'] = close_series
        features['Volume'] = volume_series.round().astype(int)
        
        # Lagged prices (useful for ML models)
        features['Close_1d'] = close_series.shift(1)
        features['Close_5d'] = close_series.shift(5)
        features['Volume_1d'] = volume_series.shift(1)
        
        # Price Action
        features['Price_Change'] = close_series.pct_change()
        features['Price_Change_1d'] = close_series.pct_change(1)
        features['Price_Change_3d'] = close_series.pct_change(3)
        features['Price_Change_5d'] = close_series.pct_change(5)
        features['High_Low_Ratio'] = high_series / low_series
        features['Open_Close_Ratio'] = open_series / close_series
        features['Close_High_Ratio'] = close_series / high_series
        features['Close_Low_Ratio'] = close_series / low_series
        features['Price_Range'] = (high_series - low_series) / close_series
        features['Gap_Up'] = (open_series / close_series.shift(1)) - 1
        features['Intraday_Return'] = (close_series - open_series) / open_series
        features['Overnight_Return'] = (open_series / close_series.shift(1)) - 1
        features['True_Range'] = np.maximum(high_series - low_series, 
                                           np.maximum(np.abs(high_series - close_series.shift(1)),
                                                    np.abs(low_series - close_series.shift(1))))
        features['Price_Position'] = (close_series - low_series) / (high_series - low_series + 0.001)
        features['Log_Return'] = np.log(close_series / close_series.shift(1))

        # Momentum
        features['RSI_14'] = talib.RSI(close, timeperiod=14)
        features['RSI_7'] = talib.RSI(close, timeperiod=7)
        features['RSI_21'] = talib.RSI(close, timeperiod=21)
        features['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        features['ROC_5'] = talib.ROC(close, timeperiod=5)
        features['ROC_10'] = talib.ROC(close, timeperiod=10)
        features['ROC_20'] = talib.ROC(close, timeperiod=20)
        features['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        features['RSI_Change'] = features['RSI_14'].diff()

        # Trend
        sma_5, sma_10, sma_20, sma_50 = talib.SMA(close, 5), talib.SMA(close, 10), talib.SMA(close, 20), talib.SMA(close, 50)
        ema_9, ema_21 = talib.EMA(close, 9), talib.EMA(close, 21)
        features['SMA_5'], features['SMA_10'], features['SMA_20'] = sma_5, sma_10, sma_20
        features['SMA_5_20_Ratio'] = sma_5 / sma_20
        features['EMA_9'], features['EMA_21'] = ema_9, ema_21
        features['Price_vs_EMA21'] = close / ema_21
        features['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        # MACD
        macd_line, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
        features['MACD'], features['MACD_Signal'], features['MACD_Histogram'] = macd_line, macd_signal, macd_hist
        features['MACD_Change'] = features['MACD'].diff()

        # Volatility
        # features['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        # features['ATR_7'] = talib.ATR(high, low, close, timeperiod=7)
        features['Std_10'] = close_series.rolling(window=10).std()
        features['Std_20'] = close_series.rolling(window=20).std()
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        features['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-6)
        features['BB_Width'] = (bb_upper - bb_lower) / (bb_middle + 1e-6)

        # Volume
        vol_sma_10 = volume_series.rolling(window=10).mean()
        vol_sma_20 = volume_series.rolling(window=20).mean()
        features['Volume_SMA_10'], features['Volume_SMA_20'] = vol_sma_10, vol_sma_20
        features['Volume_Ratio_10'] = volume / (vol_sma_10 + 1)
        features['Volume_Ratio_20'] = volume / (vol_sma_20 + 1)
        features['Volume_Change'] = volume_series.pct_change()
        features['Price_Volume'] = close_series * volume / 1000000
        features['OBV'] = talib.OBV(close, volume)
        features['SMA_Cross_5_20'] = (sma_5 > sma_20).astype(int)
        
        # =============================================================================
        # 2. ADD INFO DATA (Static context - same value for all dates)
        # =============================================================================
        info_dict = dict(zip(self.info_df.iloc[:, 0], self.info_df.iloc[:, 1]))
        
        # Market data from info (broadcast to all dates as context)
        info_features = {
            'Info_enterprise_value': info_dict.get('enterpriseValue', 0),
            'Info_market_cap': info_dict.get('marketCap', 0),
            'Info_trailing_pe': info_dict.get('trailingPE', 0),
            'Info_forward_pe': info_dict.get('forwardPE', 0),
            'Info_shares_short': info_dict.get('sharesShort', 0),
            'Info_short_ratio': info_dict.get('shortRatio', 0),
            'Info_price_to_book': info_dict.get('priceToBook', 0),
            'Info_trailing_eps': info_dict.get('trailingEps', 0),
            'Info_forward_eps': info_dict.get('forwardEps', 0),
            'Info_target_high': info_dict.get('targetHighPrice', 0),
            'Info_target_low': info_dict.get('targetLowPrice', 0),
            'Info_target_mean': info_dict.get('targetMeanPrice', 0),
            'Info_target_median': info_dict.get('targetMedianPrice', 0),
            'Info_earnings_growth': info_dict.get('earningsGrowth', 0),
            'Info_revenue_growth': info_dict.get('revenueGrowth', 0),
            'Info_current_price': info_dict.get('currentPrice', 0)
        }
        
        # Add info features as context (same value for all dates)
        for feature_name, feature_value in info_features.items():
            features[feature_name] = feature_value

        # =============================================================================
        # 4. ADD CALCULATED FUNDAMENTAL FEATURES (Time-series - specific dates only)
        # =============================================================================
        
        # Get the calculated fundamental features from your existing method
        fundamental_features_df = self.calculate_fundamental_features_timeseries(ticker)
        
        # The calculated features come with dates as columns, we need to transpose and merge properly
        # Transpose to get dates as index and features as columns
        calc_fund_df = fundamental_features_df.T
        calc_fund_df.index = pd.to_datetime(calc_fund_df.index)

        # Add 'Fund_Calc_' prefix to calculated features
        calc_fund_df.columns = [f'Fund_Calc_{col}' for col in calc_fund_df.columns]

        # =============================================================================
        # HANDLE DATE vs DATETIME MISMATCH - SMART BUSINESS DAY MATCHING
        # =============================================================================

        # Instead of exact matching, find next business day for each fundamental date
        fund_dates_business = []
        for fund_date in calc_fund_df.index:
            # For Dec 31st dates, find the NEXT available trading day
            if fund_date.month == 12 and fund_date.day == 31:
                # Start from next day and find first business day
                next_business_day = (fund_date + pd.Timedelta(days=1)) + BDay(0)
                # If that's still a holiday, keep adding business days
                while next_business_day not in features.index:
                    next_business_day = next_business_day + BDay(1)
                    # Add time component and check
                    test_datetime = next_business_day.replace(hour=9, minute=30, second=0, microsecond=0)
                    if test_datetime in features.index:
                        next_business_day = test_datetime
                        break
                business_datetime = next_business_day.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                # For other dates, use current business day logic
                next_business_day = fund_date + BDay(0)
                business_datetime = next_business_day.replace(hour=9, minute=30, second=0, microsecond=0)
            
            fund_dates_business.append(business_datetime)

        # Create new DataFrame with business day timestamps
        calc_fund_df_business = calc_fund_df.copy()
        calc_fund_df_business.index = pd.DatetimeIndex(fund_dates_business)

        # Merge with business day timestamps
        features = features.merge(calc_fund_df_business, left_index=True, right_index=True, how='left')
        features = features.sort_index(ascending=False)
        # Custom forward fill for fundamental data: for each year, use the previous Dec 31 value
        fund_calc_cols = [col for col in features.columns if col.startswith('Fund_Calc_')]
        features[fund_calc_cols] = features[fund_calc_cols].bfill().ffill()

        # =============================================================================
        # 5. ADD RATINGS DATA (Time-series - specific dates only)
        # =============================================================================
        
        if 'GradeDate' in self.ratings_df.columns:
            ratings_clean = self.ratings_df.copy()
            ratings_clean['date'] = pd.to_datetime(ratings_clean['GradeDate'])
            ratings_clean.set_index('date', inplace=True)
            
            # Add 'Rating_' prefix to rating columns (exclude GradeDate since we used it for index)
            rating_cols = [col for col in ratings_clean.columns if col != 'GradeDate']
            ratings_clean = ratings_clean[rating_cols]
            ratings_clean.columns = [f'Rating_{col}' for col in ratings_clean.columns]
            
            # Merge ratings data (NaN for missing dates - NO FORWARD FILL)
            features = features.merge(ratings_clean, left_index=True, right_index=True, how='left')

        # =============================================================================
        # 6. FINAL CLEANUP
        # =============================================================================
        
        # Clean up infinite values but DON'T forward fill fundamentals/ratings
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Only forward/backward fill technical indicators (not fundamentals or ratings)
        technical_cols = [col for col in features.columns if not (
            col.startswith('Fund_') or 
            col.startswith('Rating_') or 
            col.startswith('Info_')
        )]
        
        features[technical_cols] = features[technical_cols].ffill().bfill()
        
        # Sort by date to ensure proper order
        features = features.sort_index()

        return features

# Usage example
if __name__ == "__main__":
    fe = FeatureEngineer()
    
    # Test with a ticker
    ticker = "MMM"  # Replace with your ticker
    features_df = fe.create_features(ticker)
    features_df.to_excel('testing_data/features.xlsx', index=True)
    
    print(f"Created {len(features_df.columns)} features for {ticker}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Shape: {features_df.shape}")
    
    # Show fundamental features
    fund_cols = [col for col in features_df.columns if col.startswith('Fund_')]
    print(f"\nFundamental features ({len(fund_cols)}):")
    for col in fund_cols:
        print(f"  {col}: {features_df[col].iloc[-1]:.4f}")
