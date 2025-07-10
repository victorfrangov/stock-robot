from ib_async import *
import pandas as pd
from datetime import datetime, timedelta
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

def load_simple_ticker_list(filename='sp500_tickers.txt'):
    """Load tickers from simple text file (one per line)"""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"❌ File {filename} not found.")
        return []

# tickers = load_simple_ticker_list('sp500_tickers.txt')
tickers = ['MMM',
            'AOS',
            'ABT',
            'ABBV',
            'ACN',
            'ADBE',
            'AMD',
            'AES',
            'AFL',
            'A',
            'APD'
            ]

async def fetch_15min_data(ib, ticker, years=1):
    try:
        contract = ib.reqContractDetails(Stock(ticker, 'SMART', 'USD'))[0].contract
    except Exception as e:
        print(f"❌ {ticker}: Contract error - {e}")
        return

    bars = []
    end_time = pd.Timestamp.now(tz='US/Eastern').replace(hour=16, minute=0, second=0, microsecond=0)
    start_time = end_time - pd.Timedelta(days=365*years)
    total_weeks = int((end_time - start_time).days / 7)
    week_count = 0

    print(f"▶️ {ticker}: Starting download of 15min bars for {years} years ({total_weeks} weeks)")

    while end_time > start_time:
        end_str = end_time.strftime('%Y%m%d %H:%M:%S US/Eastern')
        week_count += 1
        print(f"   {ticker}: Requesting week {week_count}/{total_weeks} ending {end_str} ...", end="\r")
        try:
            chunk = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_str,
                durationStr='1 W',
                barSizeSetting='15 mins',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
        except Exception as e:
            print(f"\n❌ {ticker}: Error fetching data - {e}")
            break
        if not chunk:
            print(f"\n⚠️ {ticker}: No more data or hit API limit.")
            break
        bars.extend(chunk)
        end_time -= timedelta(weeks=1)
        # await asyncio.sleep(1.2)  # Respect IBKR pacing

    if bars:
        df = pd.DataFrame([{
            "date": pd.to_datetime(bar.date).tz_localize(None) if hasattr(bar, 'date') else None,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        } for bar in bars])
        df.to_parquet(f"historical_data/{ticker}_{years}y_15min.parquet", index=False)
        print(f"\n✅ {ticker}: Saved {len(df)} bars")
    else:
        print(f"\n❌ {ticker}: No data saved.")

async def main():
    ib = IB()
    ib.connect('127.0.0.1', ports[3], clientId=1)
    print(f"{'='*5}Connected{'='*5}")

    total = len(tickers)
    # Limit concurrency to avoid IBKR pacing violations
    semaphore = asyncio.Semaphore(3)  # Adjust concurrency as needed
    
    async def sem_fetch(idx, ticker):
        async with semaphore:
            await fetch_15min_data(ib, ticker)
        print(f"[{idx+1}/{total}] Finished {ticker}")

    tasks = [sem_fetch(idx, ticker) for idx, ticker in enumerate(tickers)]
    await asyncio.gather(*tasks)

    ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())