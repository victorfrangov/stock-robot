from ibapi.client import *
from ibapi.common import BarData, OrderId
from ibapi.contract import Contract
from ibapi.execution import Execution
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.wrapper import *
from ibapi.ticktype import TickTypeEnum
from decimal import Decimal
import time
import threading

port = 4002

class IBApi(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.contract = None
  
    def nextValidId(self, orderId: OrderId):
        self.orderId = orderId
        
        mycontract = Contract()
        mycontract.symbol = 'AAPL'
        mycontract.secType = "STK"
        mycontract.currency = "USD"
        mycontract.exchange = "SMART"
        mycontract.primaryExchange = "NASDAQ"
        self.reqContractDetails(self.orderId, mycontract)
            
    def nextId(self):
        self.orderId += 1
        return self.orderId

    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")

    def contractDetails(self, reqId, contractDetails):
        print(contractDetails.contract)
        
        myorder = Order()
        myorder.action = "BUY"
        myorder.orderType = "MKT"

        # self.placeOrder(self.orderId, contractDetails.contract, myorder)
        
        self.reqMarketDataType(3)
        self.reqMktData(app.nextId(), contractDetails.contract, "", False, False, [])

    def contractDetailsEnd(self, reqId: int):
        print("End of contract details")
        # self.disconnect()
        
    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"reqId: {reqId}, tickType: {TickTypeEnum.toStr(tickType)}, price: {price}, attrib: {attrib}")
    
    def tickSize(self, reqId, tickType, size):
        print(f"reqId: {reqId}, tickType: {TickTypeEnum.toStr(tickType)}, size: {size}")
        
    def historicalData(self, reqId: int, bar: BarData):
        print(reqId, bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print(f"Historical data ended for {reqId}. Started at {start} ended at {end}")
        self.cancelHistogramData(reqId)
        self.disconnect()
        
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: OrderState):
        print(f"openOrder. orderId: {orderId}, contract: {contract}, order: {order}")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: Decimal, remaining: Decimal, avgFillPrice: float, permId: OrderId, parentId: OrderId, lastFillPrice: float, clientId: OrderId, whyHeld: str, mktCapPrice: float):
        print(f"orderStatus. orderId: {orderId}, status: {status}, filled: {filled},remaining:{remaining}, avgFillPrice: {avgFillPrice}, permId: {permId}, parentId: {parentId}, lastFillPrice: {lastFillPrice}, clientId: {clientId}, whyHeld: {whyHeld}, mktCapPrice: {mktCapPrice}")
    
    def execDetails(self, reqId: OrderId, contract: Contract, execution: Execution):
        print(f"execDetails. reqId: {reqId}, contract: {contract}, execution: {execution}")
        
    def accountSummary(self, reqId: int, account: str, tag: str, value: str,currency: str):
        print("AccountSummary. ReqId:", reqId, "Account:", account,"Tag: ", tag, "Value:", value, "Currency:", currency)
    
    def accountSummaryEnd(self, reqId: int):
        print("AccountSummaryEnd. ReqId:", reqId)
    
app = IBApi()
app.connect("127.0.0.1", port, 0)
threading.Thread(target=app.run).start()
time.sleep(1)

mycontract = Contract()
mycontract.symbol = 'AAPL'
mycontract.secType = "STK"
mycontract.currency = "USD"
mycontract.exchange = "SMART"
mycontract.primaryExchange = "NASDAQ"

app.reqMktData(app.nextId(), mycontract, "", False, False, [])
# app.reqContractDetails(app.nextId(), mycontract)
# app.reqAccountSummary(app.nextId(), "All", "TotalCashValue")
# app.reqHistoricalData(app.nextId(), mycontract, "20250707 16:00:00 US/Eastern", "3 D", "1 hour", "TRADES", 1, 1, False, [])
