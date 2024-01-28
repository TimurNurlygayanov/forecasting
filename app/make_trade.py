# https://github.com/metaapi/metaapi-python-sdk/tree/main
#
# candles = await account.get_historical_candles(symbol='EURUSD', timeframe='1m',
#                                                start_time=datetime.fromisoformat('2021-05-01'), limit=1000)
# print(await connection.create_stop_sell_order(symbol='GBPUSD', volume=0.07, open_price=1.0, stop_loss=2.0,
#     take_profit=0.9, options={'comment': 'comment', 'clientId': 'TE_GBPUSD_7hyINWqAl'}))
#

import asyncio

from datetime import datetime

from metaapi_cloud_sdk import MetaApi


with open('/Users/timur.nurlygaianov/fx_api_token.txt', mode='r', encoding='utf8') as f:
    api_token = f.read().strip()


async def test():
    api = MetaApi(token=api_token)
    account = await api.metatrader_account_api.get_account(account_id='46c858ce-aeed-47fd-a9f1-d4f8c5576ac8')

    connection = account.get_rpc_connection()
    await connection.connect()

    print('account information:', await connection.get_account_information())
    print('positions:', await connection.get_positions())
    # print(await connection.get_position('1234567'))
    print('open orders:', await connection.get_orders())

    print(
        'margin required for trade',
        await connection.calculate_margin(
            {'symbol': 'GBPUSD', 'type': 'ORDER_TYPE_BUY', 'volume': 0.1, 'openPrice': 1.1}
        ),
    )

    # Get data for the ticker for the last 100 days
    now = datetime.now()
    candles = await account.get_historical_candles(symbol='EURUSD', timeframe='1d',
                                                   start_time=now, limit=100)
    print(candles)

    print('Submitting pending order')
    try:
        result = await connection.create_stop_buy_order(
            'GBPUSD', volume=0.07, open_price=1.0, stop_loss=0.9, take_profit=2.0,
            options={'comment': 'comm', 'clientId': 'TE_GBPUSD_7hyINWqAlE'}
        )
        print('Trade successful, result is ', result)
    except Exception as err:
        print('Trade failed with error:')
        print(api.format_error(err))


loop = asyncio.get_event_loop()
loop.run_until_complete(test())
loop.close()
