import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from feature_config import FeatureConfig

class DataProcessor:
    """Generates the features for the model."""
    def __init__(self):
        self.feature_config = FeatureConfig()
    
    def _find_next_trading_day(self, fund_date: pd.Timestamp, features_index: pd.DatetimeIndex) -> pd.Timestamp:
        """Find next available trading day for fundamental date"""
        exact_datetime = fund_date.replace(hour=9, minute=30, second=0)
        if exact_datetime in features_index:
            return exact_datetime
        
        if exact_datetime.month == 12 and exact_datetime.day == 31: # Will always hit this if statement as fundamentals features datastamp is always EOY
            for days_ahead in range(1, 11):
                candidate = exact_datetime + pd.Timedelta(days=days_ahead)
                if candidate in features_index:
                    return candidate
        
        next_bday = fund_date + BDay(1)
        return next_bday.replace(hour=9, minute=30, second=0)
    
    def process_data(self, ticker: str) -> pd.DataFrame:
        """Process all data and merge into final DataFrame"""
        
        # Get all feature DataFrames from FeatureConfig
        technical_features, fundamental_features, info_features, ratings_features = self.feature_config.create_all_features(ticker)
        
        # Start with technical features as base
        features = technical_features.copy()
        
        # =============================================================================
        # MERGE FUNDAMENTAL FEATURES (with business day matching)
        # =============================================================================
        if not fundamental_features.empty:
            # Map fundamental dates to trading days
            fund_dates_business = [
                self._find_next_trading_day(date, features.index) 
                for date in fundamental_features.index
            ]
            
            # Create new DataFrame with business day timestamps
            fundamental_mapped = fundamental_features.copy()
            fundamental_mapped.index = pd.DatetimeIndex(fund_dates_business)
                        
            # Merge fundamental features
            features = features.merge(fundamental_mapped, left_index=True, right_index=True, how='left')
            features = features.sort_index(ascending=False)
            
            # Forward fill fundamental data
            fund_calc_cols = [col for col in features.columns if col.startswith('Fund_Calc_')]
            features[fund_calc_cols] = features[fund_calc_cols].bfill().ffill()
            
        # =============================================================================
        # MERGE INFO FEATURES (broadcast to all dates)
        # =============================================================================
        if not info_features.empty:
            for col in info_features.columns:
                features[col] = info_features[col].iloc[0]  # Broadcast single value to all rows
        
        # =============================================================================
        # MERGE RATINGS FEATURES (with business day matching)
        # =============================================================================
        if not ratings_features.empty:
            ratings_dates_business = []
            daily_slot_counter = {}
            
            for date in ratings_features.index:
                # Get base trading day
                base_time = self._find_next_trading_day(date, features.index)
                date_key = base_time.date()
                
                # Count how many ratings we've seen for this day
                if date_key not in daily_slot_counter:
                    daily_slot_counter[date_key] = 0
                else:
                    daily_slot_counter[date_key] += 1
                
                # Calculate time slot (15-min intervals starting from base time)
                slot_number = daily_slot_counter[date_key]
                
                # Round base time up to 15-min boundary, then add slots
                current_min = base_time.minute
                remainder = current_min % 15
                if remainder == 0:
                    rounded_base = base_time
                else:
                    rounded_base = base_time + pd.Timedelta(minutes=15 - remainder)
                
                # Add 15 minutes for each additional rating
                final_time = rounded_base + pd.Timedelta(minutes=15 * slot_number)
                ratings_dates_business.append(final_time)
            
            # Add time component to ratings dates
            ratings_with_time = ratings_features.copy()
            ratings_with_time.index = pd.DatetimeIndex(ratings_dates_business)
            
            # Merge ratings
            features = features.merge(ratings_with_time, left_index=True, right_index=True, how='left')
            
            # Forward fill ratings (they persist until next rating)
            rating_cols = [col for col in features.columns if col.startswith('Rating_')]
            features[rating_cols] = features[rating_cols] # could bfill here or just not fill at all
                    
        # =============================================================================
        # FINAL CLEANUP
        # =============================================================================
        
        # Clean up infinite values
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
            features_df.to_parquet(f'processed_features/{ticker}.parquet', index=True)
            
            print(f"✅ [{idx}/{len(tickers)}] {ticker}: {len(features_df.columns)} features, {len(features_df)} rows")
            successful += 1
            
        except Exception as e:
            print(f"❌ [{idx}/{len(tickers)}] {ticker}: Error - {str(e)}")
            failed += 1
            continue
    
    print(f"\n🏁 Processing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Files saved to: processed_features/")
