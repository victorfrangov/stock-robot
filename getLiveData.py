from ib_async import *
import sys

# Gets live data of a specific stock using IKBR

#       live,  paper
ports = [7496, 7497, # tws
         4001, 4002] # gateway

ib = IB()
ib.connect('127.0.0.1', ports[3], clientId=1)

# Subscribe to live market data
stock = Stock('MSFT', 'SMART', 'USD')
options = Option('MSFT', '20250725', 505, 'C', 'SMART', "", 'USD')
stock_data = ib.reqMktData(stock, '', False, False)
options_data = ib.reqMktData(options, '', False, False)

# Print live quotes for 30 seconds
for i in range(30):
    ib.sleep(1)  # Wait 1 second
    if stock_data.last and options_data.last:
        sys.stdout.write(
            f"\rMSFT Stock: ${stock_data.last} (bid: ${stock_data.bid}, ask: ${stock_data.ask})\n"
            f"MSFT Option: ${options_data.last} (bid: ${options_data.bid}, ask: ${options_data.ask})"
        )
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.flush()
ib.disconnect()