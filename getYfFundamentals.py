import yfinance as yf

# Gets fundamentals with yf

ticker = 'MSFT'

data = yf.Ticker(ticker).get_info()
print(data)
