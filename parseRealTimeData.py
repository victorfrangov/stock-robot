from ib_async import *
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)
ticker = 'MMM'

# Request historical data
contract = ib.reqContractDetails(Stock(ticker, 'SMART', 'USD'))[0].contract
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='1 Y',
    barSizeSetting='15 mins', whatToShow='TRADES', useRTH=True)

if bars:
    df = pd.DataFrame([{
        "date": pd.to_datetime(bar.date).tz_localize(None) if hasattr(bar, 'date') else None,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume
    } for bar in bars])
    df.set_index('date', inplace=True)
    df.to_excel(f"historical_data/{ticker}.xlsx", index=False)
    print(f"\n✅ {ticker}: Saved {len(df)} bars")
    print(df.head())
else:
    print(f"\n❌ {ticker}: No data saved.")

ib.disconnect()