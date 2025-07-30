import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from ib_async import *
from feature_engineering import FeatureEngineer
from models import FeedForward, LSTM, Hybrid
import warnings
warnings.filterwarnings('ignore')

class LiveTradingBot:
    def __init__(self, model_paths, training_data_config, confidence_threshold=0.02):
        """
        Initialize the trading bot with trained models
        
        Args:
            model_paths: dict with model names and their .pth file paths
            training_data_config: The training_data dict from your notebook
            confidence_threshold: Minimum predicted return % to trigger a signal
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.scalers = training_data_config['scalers']
        self.feature_names = training_data_config['feature_names']
        self.log_transform_target = training_data_config['log_transform_target']
        self.confidence_threshold = confidence_threshold
        
        # Load trained models
        self.load_models(model_paths)
        
        # Connect to IBKR
        self.ib = IB()
        self.connect_ibkr()
        
    def load_models(self, model_paths):
        """Load all trained models"""
        print("📥 Loading trained models...")
        
        for model_name, path in model_paths.items():
            if model_name == 'FeedForward':
                model = FeedForward(input_features=len(self.feature_names))
            elif model_name == 'LSTM':
                model = LSTM(input_features=len(self.feature_names))
            elif model_name == 'Hybrid':
                model = Hybrid(input_features=len(self.feature_names))
            
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            model.to(self.device)
            self.models[model_name] = model
            print(f"✅ Loaded {model_name}")
    
    def connect_ibkr(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib.connect('127.0.0.1', 4002, clientId=999)
            print("✅ Connected to IBKR")
        except Exception as e:
            print(f"❌ IBKR connection failed: {e}")
    
    def get_live_data(self, ticker, days=30):
        """Get recent market data for a ticker"""
        try:
            # Get recent data (need historical data to calculate features)
            stock = yf.Ticker(ticker)
            data = stock.history(period=f"{days}d", interval="1d")
            
            if data.empty:
                print(f"❌ No data for {ticker}")
                return None
                
            return data
        except Exception as e:
            print(f"❌ Error getting data for {ticker}: {e}")
            return None
    
    def prepare_features(self, data, ticker):
        """Prepare features for inference"""
        try:
            # Generate features using the same FeatureEngineer
            features = self.feature_engineer.create_features(data, ticker)
            
            # Get the latest row (most recent data)
            latest_features = features.iloc[-1][self.feature_names].values.reshape(1, -1)
            
            # Scale features using the training scaler
            scaled_features = self.scalers['feature_scaler'].transform(latest_features)
            
            return torch.FloatTensor(scaled_features).to(self.device)
        except Exception as e:
            print(f"❌ Error preparing features for {ticker}: {e}")
            return None
    
    def predict_return(self, ticker):
        """Generate predictions from all models for a ticker"""
        # Get live data
        data = self.get_live_data(ticker)
        if data is None:
            return None
        
        # Prepare features
        features = self.prepare_features(data, ticker)
        if features is None:
            return None
        
        predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    # Get prediction
                    if model_name == 'LSTM':
                        # For LSTM, we need sequence data - use last 10 days
                        seq_features = []
                        for i in range(10, len(data) + 1):
                            day_data = data.iloc[i-10:i]
                            day_features = self.feature_engineer.create_features(day_data, ticker)
                            if len(day_features) > 0:
                                seq_features.append(day_features.iloc[-1][self.feature_names].values)
                        
                        if len(seq_features) >= 10:
                            seq_features = np.array(seq_features[-10:])  # Last 10 days
                            seq_features_scaled = self.scalers['feature_scaler'].transform(seq_features)
                            seq_tensor = torch.FloatTensor(seq_features_scaled).unsqueeze(0).to(self.device)
                            raw_pred = model(seq_tensor).item()
                        else:
                            continue
                    else:
                        raw_pred = model(features).item()
                    
                    # Inverse transform the prediction
                    if self.scalers['target_scaler']:
                        pred_scaled = self.scalers['target_scaler'].inverse_transform([[raw_pred]])[0][0]
                    else:
                        pred_scaled = raw_pred
                    
                    # Handle log transform
                    if self.log_transform_target:
                        pred_scaled = np.clip(pred_scaled, -10, 10)
                        final_pred = np.expm1(pred_scaled)
                    else:
                        final_pred = pred_scaled
                    
                    # Convert to percentage
                    predictions[model_name] = final_pred * 100
                    
                except Exception as e:
                    print(f"❌ Error in {model_name} prediction for {ticker}: {e}")
                    continue
        
        return predictions
    
    def generate_trading_signals(self, tickers):
        """Generate trading signals for a list of tickers"""
        signals = []
        
        print(f"🔍 Analyzing {len(tickers)} tickers...")
        
        for ticker in tickers:
            predictions = self.predict_return(ticker)
            
            if predictions:
                # Ensemble prediction (average of all models)
                avg_prediction = np.mean(list(predictions.values()))
                
                # Generate signal
                signal = self.create_signal(ticker, avg_prediction, predictions)
                if signal:
                    signals.append(signal)
        
        # Sort by strongest signals
        signals.sort(key=lambda x: abs(x['predicted_return']), reverse=True)
        
        return signals
    
    def create_signal(self, ticker, avg_prediction, all_predictions):
        """Create trading signal based on prediction"""
        confidence = abs(avg_prediction)
        
        if confidence >= self.confidence_threshold:
            return {
                'ticker': ticker,
                'action': 'BUY' if avg_prediction > 0 else 'SELL',
                'predicted_return': avg_prediction,
                'confidence': confidence,
                'model_predictions': all_predictions,
                'timestamp': datetime.now(),
                'signal_strength': 'STRONG' if confidence >= 0.05 else 'MODERATE'
            }
        
        return None
    
    def print_signals(self, signals):
        """Print trading signals in a formatted way"""
        if not signals:
            print("❌ No trading signals generated")
            return
        
        print(f"\n🎯 TRADING SIGNALS ({len(signals)} found)")
        print("=" * 80)
        
        for i, signal in enumerate(signals, 1):
            action_emoji = "📈" if signal['action'] == 'BUY' else "📉"
            strength_emoji = "🔥" if signal['signal_strength'] == 'STRONG' else "⚡"
            
            print(f"{i:2d}. {action_emoji} {signal['ticker']:5s} | "
                  f"{signal['action']:4s} | "
                  f"{signal['predicted_return']:+6.2f}% | "
                  f"{strength_emoji} {signal['signal_strength']:8s}")
            
            # Show individual model predictions
            for model, pred in signal['model_predictions'].items():
                print(f"       {model:12s}: {pred:+6.2f}%")
            print()
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("🔌 Disconnected from IBKR")

# Usage example
def main():
    # Model paths (update these to your actual paths)
    model_paths = {
        'FeedForward': 'models_limited/feedforward_l.pth',
        'LSTM': 'models_limited/lstm_l.pth', 
        'Hybrid': 'models_limited/hybrid_l.pth'
    }
    
    # You need to save your training_data config from the notebook
    # For now, I'll assume you have it saved
    import pickle
    with open('training_config.pkl', 'rb') as f:
        training_data_config = pickle.load(f)
    
    # Initialize trading bot
    bot = LiveTradingBot(model_paths, training_data_config, confidence_threshold=0.02)
    
    # Tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    # Generate signals
    signals = bot.generate_trading_signals(tickers)
    
    # Print results
    bot.print_signals(signals)
    
    # Disconnect
    bot.disconnect()

if __name__ == "__main__":
    main()