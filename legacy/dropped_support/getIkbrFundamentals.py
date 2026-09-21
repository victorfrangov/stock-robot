from ib_async import *
import os

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

ib = IB()
ib.connect('127.0.0.1', ports[3], clientId=1)
ticker = 'MSFT'

report_types = [
    'ReportsFinSummary',
    'ReportSnapshot',
    'RESC'
]
stock = Stock(ticker, 'SMART', 'USD')
os.makedirs(f'fundamentals/{ticker}', exist_ok=True)
for report in report_types:
    data = ib.reqFundamentalData(stock, report)
    with open(f'fundamentals/{ticker}/{report}.xml', 'w') as f:
        f.write(data)

ib.disconnect()