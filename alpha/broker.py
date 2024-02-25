from ib_insync import IB
from ib_insync import Stock
from ib_insync import Crypto
from ib_insync import StopOrder
from ib_insync import LimitOrder


ib_host = '127.0.0.1'
ib_port = 4001


def send_stop_order(ticker, order_type, quantity, price, stop_loss, take_profit):
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    # Define contract (replace with your contract details)
    contract = Stock(ticker, 'SMART', 'USD')

    # Place bracket order
    # https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.bracketOrder
    order_id = ib.bracketOrder(order_type, quantity, price, take_profit, stop_loss, contract=contract)

    # Disconnect from TWS or Gateway
    ib.disconnect()

    return order_id


def get_active_orders():
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    # Request a list of all open orders
    open_orders = ib.reqOpenOrders()

    # Print details of each open order
    for order in open_orders:
        print("Open Order:", order)

    # Disconnect from TWS or Gateway
    ib.disconnect()

    return open_orders


def get_pending_orders():
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    # Request real-time updates for account information
    res = ib.client.reqAllOpenOrders()
    print(res)

    # Wait for account updates
    ib.sleep(5)

    r = ib.trades()
    print(r)
    r = ib.orders()
    print(r)

    # Request all open trades
    open_trades = ib.openTrades()

    # Filter pending orders from the list of open trades
    pending_orders = [
        trade.order for trade in open_trades
        if trade.orderStatus.status in ['Pending', 'PendingSubmit']
    ]

    # Print details of pending orders
    for order in pending_orders:
        print("Pending Order:", order)

    # Disconnect from TWS or Gateway
    ib.disconnect()

    return pending_orders


def cancel_order(order):
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    ib.cancelOrder(order.orderId)

    # Disconnect from TWS or Gateway
    ib.disconnect()


def get_account_balance():
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    account_summary = ib.accountSummary(account='U14271694')

    # Print account summary
    available_amount = 0
    for item in account_summary:
        if item.tag == 'AvailableFunds':
            available_amount = float(item.value)
            print(available_amount)

    # Disconnect from TWS or Gateway
    ib.disconnect()

    return available_amount


get_account_balance()

ib = IB()
ib.connect(ib_host, ib_port, clientId=1)
# Define contract for Bitcoin (BTCUSD)
contract = Stock(symbol='AAPL', exchange='SMART', currency='USD')


# Place the bracket order for Bitcoin
order_details = ib.bracketOrder(action='SELL', quantity=1, limitPrice=180, takeProfitPrice=160, stopLossPrice=181)

for order in order_details:
    order.contract = contract
    res = ib.placeOrder(contract, order)
    print(res)

    ib.sleep(2)
    print(res.orderStatus)

# Request all open orders
all_open_orders = ib.reqAllOpenOrders()

# Print details of all open orders
for order in all_open_orders:
    print("Open Order:", order)

ib.disconnect()

get_account_balance()

test = get_pending_orders()
test2 = get_active_orders()
print(test)
