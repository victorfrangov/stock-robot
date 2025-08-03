import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    """Generates the features for the model."""
    
    def load_data(self, ticker: str) -> list[pd.DataFrame]:
        """Load all the collected data from the excel and parquet files into dataframes

        Args:
            ticker (str): Ticker symbol

        Returns:
            list[pd.DataFrame]: Returns all dataframes into a list
        """
        ohlcv_df = pd.read_excel(f'historical_data/xlsx/{ticker}.xlsx')
        fundamentals_df = pd.read_excel(f'fundamentals/{ticker}/fundamentals.xlsx')
        info_df = pd.read_excel(f'fundamentals/{ticker}/info.xlsx')
        ratings_df = pd.read_excel(f'fundamentals/{ticker}/ratings.xlsx')

        return [ohlcv_df, fundamentals_df, info_df, ratings_df]
    
    def calculate_fundamental_features(self, ticker: str) -> dict:
        """Calculate all fundamental features from parsed data
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            dict: Returns the fundamental features in a dict. Fundamentals, info and analyst ratings are all appended in the same dict.
        """
    
        # Load data directly as DataFrames
        df = self.load_data(ticker)
        fundamentals_df = df[1]
        info_df = df[2]
        ratings_df = df[3]
        
        # Get all date columns (excluding 'Source' and 'index')
        date_cols = [col for col in fundamentals_df.columns if col not in ['Source', 'index']]

        def get_fundamental(row_name):
            """Returns the value for the given fundamental at the latest date, or 0 if missing."""
            result = fundamentals_df.loc[fundamentals_df['index'] == row_name, date_cols].squeeze()
            return result.astype(np.float64).dropna() if not result.empty and pd.notnull(result.values[0]) else 0

        ebitda = get_fundamental('EBITDA')
        diluted_eps = get_fundamental('DilutedEPS')
        net_income = get_fundamental('NetIncome')
        tax_provision = get_fundamental('TaxProvision')
        operating_income = get_fundamental('OperatingIncome')
        operating_expense = get_fundamental('OperatingExpense')
        gross_profit = get_fundamental('GrossProfit')
        cost_of_revenue = get_fundamental('CostOfRevenue')
        total_revenue = get_fundamental('TotalRevenue')
    
        # Balance Sheet
        total_debt = get_fundamental('TotalDebt')
        tangible_book_value = get_fundamental('TangibleBookValue')
        stockholders_equity = get_fundamental('StockholdersEquity')
        total_assets = get_fundamental('TotalAssets')
        current_assets = get_fundamental('CurrentAssets')
        current_liabilities = get_fundamental('CurrentLiabilities')
        accounts_receivable = get_fundamental('AccountsReceivable')
        cash_and_equivalents = get_fundamental('CashAndCashEquivalents')
        shares_outstanding = get_fundamental('OrdinarySharesNumber')
    
        # Cash Flow
        free_cash_flow = get_fundamental('FreeCashFlow')
        capex = abs(get_fundamental('CapitalExpenditure'))  # Make positive
        operating_cash_flow = get_fundamental('OperatingCashFlow')
        change_in_wc = get_fundamental('ChangeInWorkingCapital')
    
        # Calculate all features
        features = {}
    
        # === CAPITAL EFFICIENCY ===
        features['return_on_assets'] = net_income / total_assets
        features['return_on_equity'] = net_income / stockholders_equity
    
        # === CASH QUALITY ===
        features['cash_conversion'] = operating_cash_flow / net_income
    
        # === CASH GENERATION QUALITY ===
        features['fcf_margin'] = free_cash_flow / total_revenue
        features['ocf_margin'] = operating_cash_flow / total_revenue
        features['fcf_conversion'] = free_cash_flow / net_income
    
        # === GROWTH & INVESTMENT ===
        features['capex_intensity'] = capex / total_revenue
        features['reinvestment_rate'] = capex / operating_cash_flow
        features['fcf_after_capex'] = operating_cash_flow - capex
    
        # === LIQUIDITY RATIOS ===
        features['current_ratio'] = current_assets / current_liabilities
        features['cash_ratio'] = cash_and_equivalents / current_liabilities
    
        # === LEVERAGE RATIOS ===
        features['debt_to_equity'] = total_debt / stockholders_equity
        features['debt_to_assets'] = total_debt / total_assets
        features['equity_ratio'] = stockholders_equity / total_assets
        features['tangible_equity_ratio'] = tangible_book_value / total_assets
    
        # === ASSET QUALITY ===
        features['asset_turnover'] = total_revenue / total_assets
        features['receivables_turnover'] = total_revenue / accounts_receivable
    
        # === GROWTH & EFFICIENCY ===
        working_capital = current_assets - current_liabilities
        features['working_capital_ratio'] = working_capital / total_assets
    
        # === PER-SHARE METRICS ===
        features['book_value_per_share'] = stockholders_equity / shares_outstanding
        features['tangible_book_per_share'] = tangible_book_value / shares_outstanding
        features['revenue_per_share'] = total_revenue / shares_outstanding
    
        # === MARGIN RATIOS ===
        features['operating_margins'] = operating_income / total_revenue
        features['ebitda_margins'] = ebitda / total_revenue
        features['gross_margins'] = gross_profit / total_revenue
        features['net_margins'] = net_income / total_revenue
        
        info_dict = dict(zip(info_df.iloc[:, 0], info_df['Value']))
                
        # Market data from info
        market_cap = info_dict.get('marketCap', 0)
        current_price = info_dict.get('currentPrice', 0)
        enterprise_value = info_dict.get('enterpriseValue', 0)
        
        # Additional ratios with market data
        features['price_to_sales'] = market_cap / total_revenue
            
        features['pe_ratio'] = current_price / diluted_eps
            
        features['ev_ebitda'] = enterprise_value / ebitda
                
        features = pd.DataFrame(list(features.items()), columns=['Key', 'Value'])
        features.to_excel('testing_data/fundamentals_features.xlsx', index=False)
        return features
    
    def calculate_fundamental_features_timeseries(self, ticker: str) -> pd.DataFrame:
        df = self.load_data(ticker)
        fundamentals_df = df[1]
        info_df = df[2]

        # Get all date columns (excluding 'Source' and 'index')
        date_cols = [col for col in fundamentals_df.columns if col not in ['Source', 'index']]

        # Helper to get a Series for each fundamental (indexed by date)
        def get_value(row_name):
            row = fundamentals_df.loc[fundamentals_df['index'] == row_name, date_cols]
            if row.empty:
                return pd.Series([np.nan] * len(date_cols), index=date_cols)
            return row.squeeze().astype(np.float64)

        info_dict = dict(zip(info_df.iloc[:, 0], info_df['Value']))
                
        # Market data from info
        market_cap = info_dict.get('marketCap', 0)
        current_price = info_dict.get('currentPrice', 0)
        enterprise_value = info_dict.get('enterpriseValue', 0)
        
        features = {
            'return_on_assets': get_value('NetIncome') / get_value('TotalAssets'),
            'return_on_equity': get_value('NetIncome') / get_value('StockholdersEquity'),
            'cash_conversion': get_value('OperatingCashFlow') / get_value('NetIncome'),
            'fcf_margin': get_value('FreeCashFlow') / get_value('TotalRevenue'),
            'ocf_margin': get_value('OperatingCashFlow') / get_value('TotalRevenue'),
            'fcf_conversion': get_value('FreeCashFlow') / get_value('NetIncome'),
            'capex_intensity': abs(get_value('CapitalExpenditure')) / get_value('TotalRevenue'),
            'reinvestment_rate': abs(get_value('CapitalExpenditure')) / get_value('OperatingCashFlow'),
            'fcf_after_capex': get_value('OperatingCashFlow') - abs(get_value('CapitalExpenditure')),
            'current_ratio': get_value('CurrentAssets') / get_value('CurrentLiabilities'),
            'cash_ratio': get_value('CashAndCashEquivalents') / get_value('CurrentLiabilities'),
            'debt_to_equity': get_value('TotalDebt') / get_value('StockholdersEquity'),
            'debt_to_assets': get_value('TotalDebt') / get_value('TotalAssets'),
            'equity_ratio': get_value('StockholdersEquity') / get_value('TotalAssets'),
            'tangible_equity_ratio': get_value('TangibleBookValue') / get_value('TotalAssets'),
            'asset_turnover': get_value('TotalRevenue') / get_value('TotalAssets'),
            'receivables_turnover': get_value('TotalRevenue') / get_value('AccountsReceivable'),
            'working_capital_ratio': (get_value('CurrentAssets') - get_value('CurrentLiabilities')) / get_value('TotalAssets'),
            'book_value_per_share': get_value('StockholdersEquity') / get_value('OrdinarySharesNumber'),
            'tangible_book_per_share': get_value('TangibleBookValue') / get_value('OrdinarySharesNumber'),
            'revenue_per_share': get_value('TotalRevenue') / get_value('OrdinarySharesNumber'),
            'operating_margins': get_value('OperatingIncome') / get_value('TotalRevenue'),
            'ebitda_margins': get_value('EBITDA') / get_value('TotalRevenue'),
            'gross_margins': get_value('GrossProfit') / get_value('TotalRevenue'),
            'net_margins': get_value('NetIncome') / get_value('TotalRevenue'),
            'market_cap': market_cap,
            'current_price': current_price,
            'enterprise_value': enterprise_value,
            'price_to_sales': market_cap / get_value('TotalRevenue'),
            'pe_ratio': current_price / get_value('DilutedEPS'),
            'ev_ebitda': enterprise_value / get_value('EBITDA')
        }
        

        # Combine all features into a DataFrame (features as rows, dates as columns)
        features_df = pd.DataFrame(features).T
        features_df.columns = pd.to_datetime(features_df.columns)
        features_df.index.name = "Feature"

        # Save to Excel
        features_df.to_excel('testing_data/fundamentals_features_timeseries.xlsx')

        return features_df
    
    def create_features(self, ticker: str) -> pd.DataFrame:
        """Compiles all the data and creates the features

        Args:
            df (pd.DataFrame): The dataframe containing the OHLCV data
            ticker (str): The stock ticker

        Returns:
            pd.DataFrame: Dataframe with compiled features (needs more processing before feeding it to a model)
        """
        df = self.load_data(ticker)
        ohlcv_df = df[0]
        fundamentals_df = df[1]
        
        # Extract OHLCV arrays
        close = ohlcv_df['close'].values.astype(np.float64)
        high = ohlcv_df['high'].values.astype(np.float64)
        low = ohlcv_df['low'].values.astype(np.float64)
        open_price = ohlcv_df['open'].values.astype(np.float64)
        volume = ohlcv_df['volume'].values.astype(np.float64)
            
        # Create series
        close_series = pd.Series(close, index=ohlcv_df.index)
        volume_series = pd.Series(volume, index=ohlcv_df.index)
        high_series = pd.Series(high, index=ohlcv_df.index)
        low_series = pd.Series(low, index=ohlcv_df.index)
        open_series = pd.Series(open_price, index=ohlcv_df.index)
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=ohlcv_df.index)

        # Raw OHLCV data
        features['Open'] = open_series
        features['High'] = high_series  
        features['Low'] = low_series
        features['Close'] = close_series
        features['Volume'] = volume_series
        
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
        
        fundamental_features = self.calculate_fundamental_features_timeseries(ticker)
        
        # Reshape fundamentals to long format
        fund_long = fundamentals_df.melt(id_vars=['Source', 'index'], var_name='date', value_name='value')
        fund_wide = fund_long.pivot_table(index='date', columns='index', values='value')
        fund_wide.index = pd.to_datetime(fund_wide.index)

        # Merge with OHLCV, forward-fill fundamentals
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
        features = ohlcv_df.merge(fund_wide, left_on='date', right_index=True, how='left')
        features = features.sort_values('date').ffill()

        # Merge ratings data
        ratings_df = df[3]
        ratings_df['date'] = pd.to_datetime(ratings_df['GradeDate'])
        ratings_df.set_index('date', inplace=True)
        features = features.merge(ratings_df, left_on='date', right_index=True, how='left')
        features = features.sort_values('date').ffill()

        # NOW add fundamental features as single values broadcast to all dates
        for feature_name, feature_value in fundamental_features.items():
            features[f'Fund_{feature_name}'] = feature_value

        return features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Usage example
if __name__ == "__main__":
    fe = FeatureEngineer()
    
    # Test with a ticker
    ticker = "MMM"  # Replace with your ticker
    features_df = fe.create_features(ticker)
    features_df.to_excel('testing_data/features.xlsx', index=False)
    
    print(f"Created {len(features_df.columns)} features for {ticker}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Shape: {features_df.shape}")
    
    # Show fundamental features
    fund_cols = [col for col in features_df.columns if col.startswith('Fund_')]
    print(f"\nFundamental features ({len(fund_cols)}):")
    for col in fund_cols:
        print(f"  {col}: {features_df[col].iloc[-1]:.4f}")
