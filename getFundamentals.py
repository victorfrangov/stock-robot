from ib_async import *
import os
from ib_fundamental import CompanyFinancials
import yfinance as yf

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

ib = IB()
ib.connect('127.0.0.1', ports[3], clientId=1)

ticker = 'MSFT'
# report_types = [
#     'ReportsFinSummary',
#     'ReportSnapshot',
#     'RESC'
# ]
# stock = Stock(ticker, 'SMART', 'USD')
# os.makedirs(f'fundamentals/{ticker}', exist_ok=True)
# for report in report_types:
#     data = ib.reqFundamentalData(stock, report)
#     with open(f'fundamentals/{ticker}/{report}.xml', 'w') as f:
#         f.write(data)

data = yf.Ticker(ticker).get_info()
print(data)
# if data is not None and not data.empty:
#     for idx, value in data.items():
#         print(f"{idx}: {value}")
# else:
#     print("No funds data available.")

# aapl = CompanyFinancials(ib, 'AAPL')

# print(aapl.revenue_ttm)

ib.disconnect()