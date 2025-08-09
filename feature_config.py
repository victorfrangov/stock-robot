import numpy as np
import pandas as pd
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
        self.ohlcv_df = pd.read_excel(f'/Volumes/storage/stock-robot-data/historical_data/xlsx/{ticker}.xlsx')
        # self.fundamentals_df = pd.read_excel(f'fundamentals/{ticker}/fundamentals.xlsx')
        # self.info_df = pd.read_excel(f'fundamentals/{ticker}/info.xlsx')
        # self.ratings_df = pd.read_excel(f'fundamentals/{ticker}/ratings.xlsx')
        
        # Clean OHLCV data
        self.ohlcv_df['date'] = pd.to_datetime(self.ohlcv_df['date'])
        self.ohlcv_df.set_index('date', inplace=True)
        self.ohlcv_df.sort_index(inplace=True)
    
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
        features['Close_15m'] = close_series.shift(1)
        features['Close_1h'] = close_series.shift(4)
        features['Close_1d'] = close_series.shift(26)
        features['Volume_15m'] = volume_series.shift(1)
        
        # Price Action
        features['Price_Change_15m'] = close_series.pct_change(1)
        features['Price_Change_1h'] = close_series.pct_change(4)
        features['Price_Change_4h'] = close_series.pct_change(16)
        features['Price_Change_1d'] = close_series.pct_change(26)
        features['High_Low_Ratio'] = high_series / low_series
        features['Price_Range'] = (high_series - low_series) / ((high_series + low_series) / 2)
        features['Intraday_Return'] = (close_series - open_series) / open_series
        features['Price_Position'] = (close_series - low_series) / (high_series - low_series + 0.001)
        
        # Momentum - Multi-timeframe (14 periods only)
        features['RSI_15m'] = talib.RSI(close, timeperiod=14)    # 14 periods = 3.5 hours
        features['RSI_1h'] = talib.RSI(close, timeperiod=56)     # 56 periods = 14 hours  
        features['RSI_4h'] = talib.RSI(close, timeperiod=224)    # 224 periods = 56 hours ≈ 1 week
        features['RSI_1d'] = talib.RSI(close, timeperiod=364)    # 364 periods = 14 trading days

        features['Williams_R_15m'] = np.clip(talib.WILLR(high, low, close, timeperiod=14), -100, 0)
        features['Williams_R_1h'] = np.clip(talib.WILLR(high, low, close, timeperiod=56), -100, 0)
        features['Williams_R_1d'] = np.clip(talib.WILLR(high, low, close, timeperiod=364), -100, 0)
        
        features['ROC_10_15m'] = talib.ROC(close, timeperiod=10)  # Changed from 1
        features['ROC_10_1h'] = talib.ROC(close, timeperiod=40)   # Changed from 4
        features['CCI_15m'] = np.clip(talib.CCI(high, low, close, timeperiod=14), -500, 500)
        features['CCI_1d'] = np.clip(talib.CCI(high, low, close, timeperiod=364), -500, 500)

        # RSI derivatives (use 15m RSI for responsiveness)
        features['RSI_Change'] = features['RSI_15m'].diff()
        features['RSI_Overbought'] = (features['RSI_15m'] > 70).astype(int)
        features['RSI_Oversold'] = (features['RSI_15m'] < 30).astype(int)
        features['RSI_Slope_15m'] = features['RSI_15m'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        features['RSI_Acceleration'] = features['RSI_Slope_15m'].diff()

        # Advanced Momentum
        features['Price_Momentum_Div'] = (features['Price_Change_1d'] > 0).astype(int) - (features['RSI_15m'] > 50).astype(int)

        # Trend
        sma_5_15m, sma_20_15m = talib.SMA(close, 5), talib.SMA(close, 20)
        ema_9_15m, ema_21_15m = talib.EMA(close, 9), talib.EMA(close, 21)
        sma_20_1d, sma_50_1d = talib.SMA(close, 520), talib.SMA(close, 1300)  # 20d, 50d
        ema_21_1d, ema_50_1d = talib.EMA(close, 546), talib.EMA(close, 1300)  # 21d, 50d
        
        features['SMA_5_20_Ratio_15m'] = sma_5_15m / sma_20_15m
        features['SMA_20_50_Ratio_1d'] = sma_20_1d / sma_50_1d
        
        features['EMA_9_15m'], features['EMA_21_15m'] = ema_9_15m, ema_21_15m
        features['EMA_21_1d'], features['EMA_50_1d'] = ema_21_1d, ema_50_1d
        
        features['Price_vs_EMA21_15m'] = close / ema_21_15m
        features['Price_vs_EMA21_1d'] = close / ema_21_1d
        
        features['Price_vs_EMA50_1d'] = close / ema_50_1d
        
        features['ADX_15m'] = talib.ADX(high, low, close, timeperiod=14)
        features['ADX_1d'] = talib.ADX(high, low, close, timeperiod=364)
        
        features['SMA_Cross_5_20_15m'] = (sma_5_15m > sma_20_15m).astype(int)
        features['SMA_Cross_20_50_1d'] = (sma_20_1d > sma_50_1d).astype(int)
        
        # MACD - Multi-timeframe
        macd_line_15m, macd_signal_15m, macd_hist_15m = talib.MACD(close, 12, 26, 9)
        macd_line_1d, macd_signal_1d, macd_hist_1d = talib.MACD(close, 312, 676, 234)  # 12d, 26d, 9d scaled
        
        features['MACD_15m'], features['MACD_Signal_15m'], features['MACD_Histogram_15m'] = macd_line_15m, macd_signal_15m, macd_hist_15m
        features['MACD_1d'], features['MACD_Signal_1d'], features['MACD_Histogram_1d'] = macd_line_1d, macd_signal_1d, macd_hist_1d
        
        features['MACD_Change_15m'] = features['MACD_15m'].diff()
        features['MACD_Bull_Cross_15m'] = ((features['MACD_15m'] > features['MACD_Signal_15m']) & 
                                (features['MACD_15m'].shift(1) <= features['MACD_Signal_15m'].shift(1))).astype(int)
        features['MACD_Bear_Cross_15m'] = ((features['MACD_15m'] < features['MACD_Signal_15m']) & 
                                (features['MACD_15m'].shift(1) >= features['MACD_Signal_15m'].shift(1))).astype(int)
        
        features['MACD_Position_15m'] = (features['MACD_15m'] > features['MACD_Signal_15m']).astype(int)
        features['MACD_Position_1d'] = (features['MACD_1d'] > features['MACD_Signal_1d']).astype(int)

        # Volatility - Multi-timeframe
        features['ATR_14_15m'] = talib.ATR(high, low, close, timeperiod=14)
        features['ATR_14_1d'] = talib.ATR(high, low, close, timeperiod=364)  # 14 trading days
        
        features['ATR_Ratio_15m'] = features['ATR_14_15m'] / close_series
        features['ATR_Ratio_1d'] = features['ATR_14_1d'] / close_series
        
        features['Std_10_15m'] = close_series.rolling(window=10).std()
        features['Std_10_1d'] = close_series.rolling(window=520).std()
        
        features['Std_20_15m'] = close_series.rolling(window=20).std()
        features['Std_20_1d'] = close_series.rolling(window=520).std()  # 20 trading days

        # Bollinger Bands - Multi-timeframe
        bb_upper_15m, bb_middle_15m, bb_lower_15m = talib.BBANDS(close, 20, 2, 2)
        bb_upper_1d, bb_middle_1d, bb_lower_1d = talib.BBANDS(close, 520, 2, 2)  # 20 trading days
        
        features['BB_Position_15m'] = np.clip((close - bb_lower_15m) / (bb_upper_15m - bb_lower_15m + 1e-6), -3, 3)
        features['BB_Position_1d'] = np.clip((close - bb_lower_1d) / (bb_upper_1d - bb_lower_1d + 1e-6), -3, 3)
        
        features['BB_Width_15m'] = (bb_upper_15m - bb_lower_15m) / (bb_middle_15m + 1e-6)
        features['BB_Width_1d'] = (bb_upper_1d - bb_lower_1d) / (bb_middle_1d + 1e-6)
        
        features['BB_Squeeze_15m'] = (features['BB_Width_15m'] < features['BB_Width_15m'].rolling(20).quantile(0.2)).astype(int)
        features['BB_Squeeze_1d'] = (features['BB_Width_1d'] < features['BB_Width_1d'].rolling(520).quantile(0.2)).astype(int)

        # Volume - Multi-timeframe
        vol_sma_10_15m = volume_series.rolling(window=10).mean()
        vol_sma_20_15m = volume_series.rolling(window=20).mean()
        vol_sma_20_1d = volume_series.rolling(window=520).mean()
        
        features['Volume_Ratio_10_15m'] = volume / (vol_sma_10_15m + 1)
        features['Volume_Ratio_20_15m'] = volume / (vol_sma_20_15m + 1)
        features['Volume_Ratio_20_1d'] = volume / (vol_sma_20_1d + 1)

        features['Volume_Spike_15m'] = (features['Volume_Ratio_20_15m'] > 2.0).astype(int)
        features['Volume_Spike_1d'] = (features['Volume_Ratio_20_1d'] > 1.5).astype(int)

        features['Volume_Change'] = np.clip(volume_series.pct_change().replace([np.inf, -np.inf], np.nan), -0.99, 10.0)
        features['OBV'] = talib.OBV(close, volume)

        # Risk Metrics - Multi-timeframe
        features['Max_Drawdown_4h'] = (close_series / close_series.rolling(16).max() - 1) * 100
        features['Max_Drawdown_20d'] = (close_series / close_series.rolling(520).max() - 1) * 100
        
        features['Underwater_Periods_4h'] = (close_series < close_series.rolling(16).max()).astype(int).rolling(16).sum()
        features['Underwater_Days'] = (close_series < close_series.rolling(520).max()).astype(int).rolling(520).sum()

        # Sharpe approximation (corrected for 15-min data)
        features['Sharpe_Approx_4h'] = np.clip((features['Price_Change_15m'].rolling(16).mean() / (features['Price_Change_15m'].rolling(16).std() + 1e-8)) * np.sqrt(96*4), -50, 50)
        features['Sharpe_Approx_20d'] = np.clip((features['Price_Change_15m'].rolling(520).mean() / (features['Price_Change_15m'].rolling(520).std() + 1e-8)) * np.sqrt(96*252), -50, 50)

        # Recovery Patterns - Multi-timeframe
        peak_4h = close_series.rolling(16).max()
        peak_20d = close_series.rolling(520).max()
        
        features['Distance_From_Peak_4h'] = (close_series / peak_4h - 1) * 100
        features['Distance_From_Peak_20d'] = (close_series / peak_20d - 1) * 100
        
        features['Periods_Since_Peak_4h'] = features.groupby((peak_4h != peak_4h.shift(1)).cumsum()).cumcount()
        features['Days_Since_Peak'] = features.groupby((peak_20d != peak_20d.shift(1)).cumsum()).cumcount()

        return features.replace([np.inf, -np.inf], np.nan)
    
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
        # fundamental_features = self.create_fundamental_features()
        # info_features = self.create_info_features()
        # ratings_features = self.create_ratings_features()
        
        # return technical_features, fundamental_features, info_features, ratings_features
        return technical_features, None, None, None