import schedule
import time
from live_inference import LiveTradingBot
import json
from datetime import datetime

def run_daily_analysis():
    """Run daily trading analysis"""
    print(f"\n🚀 Starting daily analysis at {datetime.now()}")
    
    # Your configuration
    model_paths = {
        'FeedForward': 'models_limited/feedforward_l.pth',
        'LSTM': 'models_limited/lstm_l.pth',
        'Hybrid': 'models_limited/hybrid_l.pth'
    }
    
    # Load training config
    import pickle
    with open('training_config.pkl', 'rb') as f:
        training_data_config = pickle.load(f)
    
    # Initialize bot
    bot = LiveTradingBot(model_paths, training_data_config)
    
    # Analyze S&P 500 or your watchlist
    with open('sp500_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f.readlines()[:50]  # First 50 for testing
    
    # Generate signals
    signals = bot.generate_trading_signals(tickers)
    
    # Save signals to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    with open(f'signals/signals_{timestamp}.json', 'w') as f:
        json.dump(signals, f, indent=2, default=str)
    
    # Print signals
    bot.print_signals(signals)
    
    bot.disconnect()

# Schedule the analysis
schedule.every().day.at("09:00").do(run_daily_analysis)  # 9 AM daily
schedule.every().monday.at("16:30").do(run_daily_analysis)  # After market close

print("📅 Scheduler started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute