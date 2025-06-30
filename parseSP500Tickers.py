import pandas as pd

# Alternative method using pandas (easier)
def get_sp500_tickers_pandas():
    """
    Get S&P 500 tickers using pandas (simpler method)
    """
    try:
        # Read tables from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        
        # The first table contains the S&P 500 companies
        sp500_table = tables[0]
        
        # Extract tickers and company names
        tickers = sp500_table['Symbol'].to_list()
        
        return tickers
        
    except Exception as e:
        print(f"❌ Error with pandas method: {e}")
        return []

tickers = get_sp500_tickers_pandas()

# Display sample tickers
if tickers:
    # Save to list for use in your notebook
    print(f"\n✅ Ready to use {len(tickers)} tickers in your model!")
    
    with open('sp500_tickers.txt', 'w') as f:
        for ticker in tickers:
            f.write(ticker + "\n")
    
    print(f"💾 Saved to 'sp500_tickers.txt'")
    
else:
    ticker_list = []