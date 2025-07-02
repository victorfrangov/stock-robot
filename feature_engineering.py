import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    """Generates the features for the model."""
    def create_features(self, df: pd.DataFrame, ticker) -> pd.DataFrame:
        df_copy = df.copy()
        
        if isinstance(df_copy.columns, pd.MultiIndex):
            # Extract data correctly from multi-index
            close = df_copy['Close'][ticker].values.astype(np.float64)
            high = df_copy['High'][ticker].values.astype(np.float64)
            low = df_copy['Low'][ticker].values.astype(np.float64)
            open = df_copy['Open'][ticker].values.astype(np.float64)
            volume = df_copy['Volume'][ticker].values.astype(np.float64)
            
            # Create clean DataFrame for calculations
            df = pd.DataFrame({
                'Open': open,
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
            open = df['Open'].values.astype(np.float64)
            volume = df['Volume'].values.astype(np.float64)
            
        close_series = pd.Series(close, index=df.index)
        volume_series = pd.Series(volume, index=df.index)
        high_series = pd.Series(high, index=df.index)
        low_series = pd.Series(low, index=df.index)
        open_series = pd.Series(open, index=df.index)
        features = pd.DataFrame(index=df_copy.index)

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
        features['MAC_Change'] = features['MACD'].diff()

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
        # features['SMA_Cross_10_50'] = (sma_10 > sma_50).astype(int)
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
