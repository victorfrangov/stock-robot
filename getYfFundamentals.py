import yfinance as yf

ticker = 'MSFT'

data = yf.Ticker(ticker).get_info()
print(data)
