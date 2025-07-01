import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    """Generates the 48 features for the model."""
    def create_features(self, df: pd.DataFrame, ticker) -> pd.DataFrame:
        df_copy = df.copy()
        
        if isinstance(df_copy.columns, pd.MultiIndex):
            # Extract data correctly from multi-index
            close = df_copy['Close'][ticker].values.astype(np.float64)
            high = df_copy['High'][ticker].values.astype(np.float64)
            low = df_copy['Low'][ticker].values.astype(np.float64)
            open_price = df_copy['Open'][ticker].values.astype(np.float64)
            volume = df_copy['Volume'][ticker].values.astype(np.float64)
            
            # Create clean DataFrame for calculations
            df = pd.DataFrame({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            }, index=df_copy.index)
        else:
            # Single stock download
            df = df_copy.copy()
            close = df['Close'].values.astype(np.float64)
            high = df['High'].values.astype(np.float64)
            low = df['Low'].values.astype(np.float64)
            open_price = df['Open'].values.astype(np.float64)
            volume = df['Volume'].values.astype(np.float64)

        features = pd.DataFrame(index=df_copy.index)

        # Price Action (12)
        features['Price_Change'] = df['Close'].pct_change()
        features['High_Low_Ratio'] = df['High'] / df['Low']
        features['Open_Close_Ratio'] = df['Open'] / df['Close']
        features['Close_High_Ratio'] = df['Close'] / df['High']
        features['Close_Low_Ratio'] = df['Close'] / df['Low']
        features['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        features['Gap_Up'] = (df['Open'] / df['Close'].shift(1)) - 1
        features['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
        features['Overnight_Return'] = (df['Open'] / df['Close'].shift(1)) - 1
        features['True_Range'] = np.maximum(df['High'] - df['Low'], 
                                           np.maximum(np.abs(df['High'] - df['Close'].shift(1)),
                                                    np.abs(df['Low'] - df['Close'].shift(1))))
        features['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)
        features['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Momentum (10)
        features['RSI_14'] = talib.RSI(close, timeperiod=14)
        features['RSI_7'] = talib.RSI(close, timeperiod=7)
        features['RSI_21'] = talib.RSI(close, timeperiod=21)
        features['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        features['ROC_5'] = talib.ROC(close, timeperiod=5)
        features['ROC_10'] = talib.ROC(close, timeperiod=10)
        features['ROC_20'] = talib.ROC(close, timeperiod=20)
        features['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        features['Momentum_10'] = talib.MOM(close, timeperiod=10)
        features['Momentum_20'] = talib.MOM(close, timeperiod=20)

        # Trend (10)
        sma_5, sma_10, sma_20, sma_50 = talib.SMA(close, 5), talib.SMA(close, 10), talib.SMA(close, 20), talib.SMA(close, 50)
        ema_9, ema_21 = talib.EMA(close, 9), talib.EMA(close, 21)
        features['SMA_5'], features['SMA_10'], features['SMA_20'] = sma_5, sma_10, sma_20
        features['Price_vs_SMA20'] = close / sma_20
        features['Price_vs_SMA50'] = close / sma_50
        features['SMA_5_20_Ratio'] = sma_5 / sma_20
        features['EMA_9'], features['EMA_21'] = ema_9, ema_21
        features['Price_vs_EMA21'] = close / ema_21
        features['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        # MACD (3)
        macd_line, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
        features['MACD'], features['MACD_Signal'], features['MACD_Histogram'] = macd_line, macd_signal, macd_hist

        # Volatility (6)
        features['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        features['ATR_7'] = talib.ATR(high, low, close, timeperiod=7)
        features['Std_10'] = df['Close'].rolling(window=10).std()
        features['Std_20'] = df['Close'].rolling(window=20).std()
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
        features['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-6)
        features['BB_Width'] = (bb_upper - bb_lower) / (bb_middle + 1e-6)

        # Volume (7)
        vol_sma_10 = df['Volume'].rolling(window=10).mean()
        vol_sma_20 = df['Volume'].rolling(window=20).mean()
        features['Volume_SMA_10'], features['Volume_SMA_20'] = vol_sma_10, vol_sma_20
        features['Volume_Ratio_10'] = volume / (vol_sma_10 + 1)
        features['Volume_Ratio_20'] = volume / (vol_sma_20 + 1)
        features['Volume_Change'] = df['Volume'].pct_change()
        features['Price_Volume'] = df['Close'] * df['Volume'] / 1000000
        features['OBV'] = talib.OBV(close, volume)
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
