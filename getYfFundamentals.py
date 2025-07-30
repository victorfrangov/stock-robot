import yfinance as yf

# Gets fundamentals with yf

ticker = 'MSFT'

data = yf.Ticker(ticker).get_balance_sheet()
print(data)

with open('testing_data/get_upgrade_downgrades.txt', 'w') as f:
    f.write(data.to_string())