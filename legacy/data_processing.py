import pandas as pd
from feature_config import FeatureConfig

class DataProcessor:
    """Generates the features for the model."""
    def __init__(self):
        self.feature_config = FeatureConfig()
    
    def process_data(self, ticker: str) -> pd.DataFrame:
        """Process all data and merge into final DataFrame"""
        
        # Get all feature DataFrames from FeatureConfig
        technical_features = self.feature_config.create_features(ticker)
        
        # Start with technical features as base
        features = technical_features.copy()
        # features = features.iloc[1300:] # Remove null values before saving the files to save on space.
        features = features.sort_index()
        
        print(f"\nFinal dataset: {features.shape}")
        
        return features

# Usage example
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load tickers
    with open('sp500_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]
        
    successful = 0
    failed = 0
    
    for idx, ticker in enumerate(tickers, 1):
        try:
            print(f"[{idx}/{len(tickers)}] Processing {ticker}...")
            features_df = processor.process_data(ticker)
            
            # Save to parquet
            features_df.to_parquet(f'processed_features_cleaned_compact/{ticker}.parquet', index=True)
            
            print(f"✅ [{idx}/{len(tickers)}] {ticker}: {len(features_df.columns)} features, {len(features_df)} rows")
            successful += 1
            
        except Exception as e:
            print(f"❌ [{idx}/{len(tickers)}] {ticker}: Error - {str(e)}")
            failed += 1
            continue
    
    print(f"\n🏁 Processing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
