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
        ohlcv_df = pd.read_parquet(f'historical_data/parquet/{ticker}.parquet')
        fundamentals_df = pd.read_excel(f'fundamentals/{ticker}/fundamentals.xlsx')
        info_df = pd.read_excel(f'fundamentals/{ticker}/info.xlsx')
        ratings_df = pd.read_excel(f'fundamentals/{ticker}/ratings.xlsx')

        return [ohlcv_df, fundamentals_df, info_df, ratings_df]
    
    def calculate_fundamental_features(self, ticker: str) -> dict:
        """Calculate all fundamental features from parsed data"""
    
        # Load data directly as DataFrames
        data_list = self.load_data(ticker)
        fundamentals_df = data_list[1]  # fundamentals.xlsx
        info_df = data_list[2]  # info.xlsx
        ratings_df = data_list[3] # ratings.xlsx
    
        # Convert fundamentals to a simple lookup (first column = row names, second column = values)
        if fundamentals_df.shape[1] >= 2:
            fund_data = dict(zip(fundamentals_df.iloc[:, 0], fundamentals_df.iloc[:, 1]))
        else:
            fund_data = {}
    
        # Convert info to lookup
        if info_df.shape[1] >= 2:
            info_data = dict(zip(info_df.iloc[:, 0], info_df.iloc[:, 1]))
        else:
            info_data = {}
    
        # Helper function to safely get values
        def get_value(data_dict, key, default=0):
            return data_dict.get(key, default) if data_dict.get(key) is not None else default
    
        # Extract values directly from DataFrames
        # Income Statement
        ebitda = get_value(fund_data, 'EBITDA')
        diluted_eps = get_value(fund_data, 'DilutedEPS')
        net_income = get_value(fund_data, 'NetIncome')
        tax_provision = get_value(fund_data, 'TaxProvision')
        operating_income = get_value(fund_data, 'OperatingIncome')
        operating_expense = get_value(fund_data, 'OperatingExpense')
        gross_profit = get_value(fund_data, 'GrossProfit')
        cost_of_revenue = get_value(fund_data, 'CostOfRevenue')
        total_revenue = get_value(fund_data, 'TotalRevenue')
    
        # Balance Sheet
        total_debt = get_value(fund_data, 'TotalDebt')
        tangible_book_value = get_value(fund_data, 'TangibleBookValue')
        stockholders_equity = get_value(fund_data, 'StockholdersEquity')
        total_assets = get_value(fund_data, 'TotalAssets')
        current_assets = get_value(fund_data, 'CurrentAssets')
        current_liabilities = get_value(fund_data, 'CurrentLiabilities')
        inventory = get_value(fund_data, 'Inventory')
        accounts_receivable = get_value(fund_data, 'AccountsReceivable')
        cash_and_equivalents = get_value(fund_data, 'CashAndCashEquivalents')
        shares_outstanding = get_value(fund_data, 'OrdinarySharesNumber')
    
        # Cash Flow
        free_cash_flow = get_value(fund_data, 'FreeCashFlow')
        capex = abs(get_value(fund_data, 'CapitalExpenditure'))  # Make positive
        operating_cash_flow = get_value(fund_data, 'OperatingCashFlow')
        change_in_wc = get_value(fund_data, 'ChangeInWorkingCapital')
    
        # Calculate all features
        features = {}
    
        # === CAPITAL EFFICIENCY ===
        features['roa'] = net_income / total_assets if total_assets > 0 else 0
        features['roe'] = net_income / stockholders_equity if stockholders_equity > 0 else 0
    
        # ROIC calculation (simplified)
        ebit = operating_income  # Approximation
        tax_rate = tax_provision / (net_income + tax_provision) if (net_income + tax_provision) > 0 else 0
        invested_capital = stockholders_equity + total_debt  # Simplified
        features['roic'] = (ebit * (1 - tax_rate)) / invested_capital if invested_capital > 0 else 0
    
        # === CASH QUALITY ===
        features['cash_conversion'] = operating_cash_flow / net_income if net_income > 0 else 0
    
        # === CASH GENERATION QUALITY ===
        features['fcf_margin'] = free_cash_flow / total_revenue if total_revenue > 0 else 0
        features['ocf_margin'] = operating_cash_flow / total_revenue if total_revenue > 0 else 0
        features['fcf_conversion'] = free_cash_flow / net_income if net_income > 0 else 0
    
        # === GROWTH & INVESTMENT ===
        features['capex_intensity'] = capex / total_revenue if total_revenue > 0 else 0
        features['reinvestment_rate'] = capex / operating_cash_flow if operating_cash_flow > 0 else 0
        features['fcf_after_capex'] = operating_cash_flow - capex
    
        # === LIQUIDITY RATIOS ===
        features['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 0
        features['cash_ratio'] = cash_and_equivalents / current_liabilities if current_liabilities > 0 else 0
    
        # === LEVERAGE RATIOS ===
        features['debt_to_equity'] = total_debt / stockholders_equity if stockholders_equity > 0 else 0
        features['debt_to_assets'] = total_debt / total_assets if total_assets > 0 else 0
        features['equity_ratio'] = stockholders_equity / total_assets if total_assets > 0 else 0
        features['tangible_equity_ratio'] = tangible_book_value / total_assets if total_assets > 0 else 0
    
        # === ASSET QUALITY ===
        features['asset_turnover'] = total_revenue / total_assets if total_assets > 0 else 0
        features['receivables_turnover'] = total_revenue / accounts_receivable if accounts_receivable > 0 else 0
        features['inventory_turnover'] = cost_of_revenue / inventory if inventory > 0 else 0
    
        # === GROWTH & EFFICIENCY ===
        working_capital = current_assets - current_liabilities
        features['working_capital_ratio'] = working_capital / total_assets if total_assets > 0 else 0
    
        # === PER-SHARE METRICS ===
        features['book_value_per_share'] = stockholders_equity / shares_outstanding if shares_outstanding > 0 else 0
        features['tangible_book_per_share'] = tangible_book_value / shares_outstanding if shares_outstanding > 0 else 0
        features['revenue_per_share'] = total_revenue / shares_outstanding if shares_outstanding > 0 else 0
    
        # === MARGIN RATIOS ===
        features['operating_margins'] = operating_income / total_revenue if total_revenue > 0 else 0
        features['ebitda_margins'] = ebitda / total_revenue if total_revenue > 0 else 0
        features['gross_margins'] = gross_profit / total_revenue if total_revenue > 0 else 0
        features['net_margins'] = net_income / total_revenue if total_revenue > 0 else 0
    
        # === EFFICIENCY METRICS ===
        features['return_on_assets'] = features['roa']  # Alias
        features['return_on_equity'] = features['roe']  # Alias
    
        info_dict = dict(zip(info_df.iloc[:, 0], info_df['Value']))
                
        # Market data from info
        market_cap = info_dict.get('marketCap', 0)
        current_price = info_dict.get('currentPrice', 0)
        enterprise_value = info_dict.get('enterpriseValue', 0)
        
        # Additional ratios with market data
        features['price_to_sales'] = market_cap / total_revenue if total_revenue > 0 and market_cap > 0 else 0
            
        features['pe_ratio'] = current_price / diluted_eps if current_price > 0 and diluted_eps > 0 else 0
            
        features['ev_ebitda'] = enterprise_value / ebitda if enterprise_value > 0 and ebitda > 0 else 0
        
        return features
    
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
        
        try:
            fundamental_features = self.calculate_fundamental_features(ticker)
            
            # Add fundamental features as single values broadcast to all dates
            for feature_name, feature_value in fundamental_features.items():
                features[f'Fund_{feature_name}'] = feature_value
                
        except Exception as e:
            print(f"⚠️ Could not calculate fundamental features for {ticker}: {e}")
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')

# Usage example
if __name__ == "__main__":
    fe = FeatureEngineer()
    
    # Test with a ticker
    ticker = "AAPL"  # Replace with your ticker
    features_df = fe.create_features(ticker)
    
    print(f"Created {len(features_df.columns)} features for {ticker}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Shape: {features_df.shape}")
    
    # Show fundamental features
    fund_cols = [col for col in features_df.columns if col.startswith('Fund_')]
    print(f"\nFundamental features ({len(fund_cols)}):")
    for col in fund_cols:
        print(f"  {col}: {features_df[col].iloc[-1]:.4f}")
