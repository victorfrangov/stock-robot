import pandas as pd
import numpy as np
import talib

class FeatureConfig:
    def __init__(self):
        """Initialize variables to store the data"""
        self.ticker = None
        self.ohlcv_df = None
        self.fundamentals_df = None
        self.info_df = None
        self.ratings_df = None
            
    def load_data(self, ticker: str) -> None:
        """Load all data from files"""
        self.ticker = ticker
        self.ohlcv_df = pd.read_excel(f'historical_data/xlsx/{ticker}.xlsx')
        self.fundamentals_df = pd.read_excel(f'fundamentals/{ticker}/fundamentals.xlsx')
        self.info_df = pd.read_excel(f'fundamentals/{ticker}/info.xlsx')
        self.ratings_df = pd.read_excel(f'fundamentals/{ticker}/ratings.xlsx')
        
        # Clean OHLCV data
        self.ohlcv_df['date'] = pd.to_datetime(self.ohlcv_df['date'])
        self.ohlcv_df.set_index('date', inplace=True)
    
    def create_technical_features(self) -> pd.DataFrame:
        """Create all technical features from OHLCV data"""
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
        features['Price_Change_1d'] = close_series.pct_change(1)
        features['Price_Change_3d'] = close_series.pct_change(3)
        features['Price_Change_5d'] = close_series.pct_change(5)
        features['High_Low_Ratio'] = high_series / low_series
        features['Price_Range'] = (high_series - low_series) / ((high_series + low_series) / 2)
        features['Intraday_Return'] = (close_series - open_series) / open_series
        features['Overnight_Return'] = (open_series / close_series.shift(1)) - 1
        features['True_Range'] = np.maximum(high_series - low_series, 
                                           np.maximum(np.abs(high_series - close_series.shift(1)),
                                                    np.abs(low_series - close_series.shift(1))))
        features['Price_Position'] = (close_series - low_series) / (high_series - low_series + 0.001)
        features['Log_Return'] = np.log(close_series / close_series.shift(1))

        # Momentum
        features['RSI_14'] = talib.RSI(close, timeperiod=14)
        features['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        features['ROC_10'] = talib.ROC(close, timeperiod=10)
        features['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        features['RSI_Change'] = features['RSI_14'].diff()
        features['RSI_Overbought'] = (features['RSI_14'] > 70).astype(int)
        features['RSI_Oversold'] = (features['RSI_14'] < 30).astype(int)

        # Advanced Momentum
        features['Price_Momentum_Div'] = (features['Price_Change_5d'] > 0).astype(int) - (features['RSI_14'] > 50).astype(int)

        # Multi-timeframe RSI
        features['RSI_Slope'] = features['RSI_14'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        features['RSI_Acceleration'] = features['RSI_Slope'].diff()

        # Trend
        sma_5, sma_20 = talib.SMA(close, 5), talib.SMA(close, 20)
        ema_9, ema_21 = talib.EMA(close, 9), talib.EMA(close, 21)
        features['SMA_5_20_Ratio'] = sma_5 / sma_20
        features['EMA_9'], features['EMA_21'] = ema_9, ema_21
        features['Price_vs_EMA21'] = close / ema_21
        features['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        features['SMA_Cross_5_20'] = (sma_5 > sma_20).astype(int)

        # MACD
        macd_line, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
        features['MACD'], features['MACD_Signal'], features['MACD_Histogram'] = macd_line, macd_signal, macd_hist
        features['MACD_Change'] = features['MACD'].diff()
        features['MACD_Bull_Cross'] = ((features['MACD'] > features['MACD_Signal']) & 
                              (features['MACD'].shift(1) <= features['MACD_Signal'].shift(1))).astype(int)
        features['MACD_Bear_Cross'] = ((features['MACD'] < features['MACD_Signal']) & 
                              (features['MACD'].shift(1) >= features['MACD_Signal'].shift(1))).astype(int)
        features['MACD_Position'] = (features['MACD'] > features['MACD_Signal']).astype(int)

        # Volatility
        features['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        features['ATR_Ratio'] = features['ATR_14'] / close_series
        features['Std_10'] = close_series.rolling(window=10).std()
        features['Std_20'] = close_series.rolling(window=20).std()
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        features['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-6)
        features['BB_Width'] = (bb_upper - bb_lower) / (bb_middle + 1e-6)
        features['BB_Squeeze'] = (features['BB_Width'] < features['BB_Width'].rolling(20).quantile(0.2)).astype(int)

        # Volume
        vol_sma_10 = volume_series.rolling(window=10).mean()
        vol_sma_20 = volume_series.rolling(window=20).mean()
        features['Volume_Ratio_10'] = volume / (vol_sma_10 + 1)
        features['Volume_Ratio_20'] = volume / (vol_sma_20 + 1)
        features['Volume_Change'] = volume_series.pct_change()
        features['OBV'] = talib.OBV(close, volume)
        features['Volume_Spike'] = (features['Volume_Ratio_20'] > 2.0).astype(int)
        
        # Risk Metrics
        features['Max_Drawdown_20'] = (close_series / close_series.rolling(20).max() - 1) * 100
        features['Underwater_Days'] = (close_series < close_series.rolling(20).max()).astype(int).rolling(20).sum()
        features['Sharpe_Approx'] = (features['Price_Change_1d'].rolling(20).mean() / (features['Price_Change_1d'].rolling(20).std() + 1e-8)) * np.sqrt(252)

        # Recovery Patterns
        peak_20 = close_series.rolling(20).max()
        features['Distance_From_Peak'] = (close_series / peak_20 - 1) * 100
        features['Days_Since_Peak'] = features.groupby((peak_20 != peak_20.shift(1)).cumsum()).cumcount()
        
        return features
    
    def create_fundamental_features(self) -> pd.DataFrame:
        """Calculate fundamental features and return as time series"""
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
        
        # Create DataFrame with features as rows, dates as columns
        funds_features = pd.DataFrame(features).T
        funds_features.columns = pd.to_datetime(funds_features.columns)
        funds_features.index.name = "Feature"
        
        # Transpose to get dates as index, features as columns (time series format)
        funds_features_ts = funds_features.T
        funds_features_ts.columns = [f'Fund_Calc_{col}' for col in funds_features_ts.columns]
        
        return funds_features_ts
    
    def create_info_features(self) -> pd.DataFrame:
        """Create info features as single-row DataFrame"""
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
        
        # Return as single-row DataFrame
        return pd.DataFrame([info_features])
    
    def create_ratings_features(self) -> pd.DataFrame:
        """Create ratings features as time series"""
        if 'GradeDate' not in self.ratings_df.columns:
            # Return empty DataFrame if no ratings data
            return pd.DataFrame()
        
        ratings_clean = self.ratings_df.copy()
        ratings_clean['date'] = pd.to_datetime(ratings_clean['GradeDate'])
        ratings_clean.set_index('date', inplace=True)
        
        # Add 'Rating_' prefix to rating columns (exclude GradeDate since we used it for index)
        rating_cols = [col for col in ratings_clean.columns if col != 'GradeDate']
        ratings_clean.columns = [f'Rating_{col}' if col in rating_cols else col for col in ratings_clean.columns]
        
        return ratings_clean
    
    def create_all_features(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create all features and return 3 separate DataFrames
        
        Returns:
            tuple: (technical_features, fundamental_features, info_features, ratings_features)
        """
        self.load_data(ticker)
        
        technical_features = self.create_technical_features()
        fundamental_features = self.create_fundamental_features()
        info_features = self.create_info_features()
        ratings_features = self.create_ratings_features()
        
        return technical_features, fundamental_features, info_features, ratings_features