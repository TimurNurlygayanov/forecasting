import time

from alpha.broker import get_pending_orders
from alpha.broker import cancel_order


orders = get_pending_orders()
time.sleep(5)

for order in orders:
    cancel_order(order)

get_pending_orders()
