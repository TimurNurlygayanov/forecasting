import time

from ib_insync import IB
from ib_insync import Stock
from ib_insync import Crypto
from ib_insync import StopOrder
from ib_insync import LimitOrder


ib_host = '127.0.0.1'
ib_port = 4001


def send_stop_order(ticker, order_type, quantity, price, delta_price, stop_loss, take_profit):
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    # Define contract
    contract = Stock(symbol=ticker, exchange='SMART', currency='USD')

    # Bracket order
    # https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.bracketOrder
    order_details = ib.bracketOrder(
        action=order_type, quantity=quantity, stopPrice=delta_price,
        limitPrice=price, takeProfitPrice=take_profit, stopLossPrice=stop_loss
    )

    for order in order_details:
        order.contract = contract
        res = ib.placeOrder(contract, order)
        print(res)

        ib.sleep(1)
        print(res.orderStatus)

    # Request all open orders
    all_open_orders = ib.reqAllOpenOrders()

    print('* ' * 20)
    print('All open orders:')
    for order in all_open_orders:
        print(order)

    # Disconnect from TWS or Gateway
    ib.disconnect()
    time.sleep(1)  # give some time to properly disconnect


def get_active_trades():
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    # Request a list of all open orders
    open_trades = ib.trades()

    # Print details of each open order
    for trade in open_trades:
        print("Open Order:", trade)

    # Disconnect from TWS or Gateway
    ib.disconnect()

    return open_trades


def get_pending_orders():
    # Connect to TWS or Gateway
    ib = IB()
    ib.connect(ib_host, ib_port, clientId=1)

    # Request all open trades
    pending_orders = ib.openOrders()

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

    ib.cancelOrder(order)
    ib.sleep(2)

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
    time.sleep(1)  # give some time to properly diconnect

    return available_amount


# get_account_balance()
# get_active_trades()
