import json
import warnings

warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings("ignore", message="urllib3")


import asyncio
import datetime

from bot.utils import get_data
from bot.utils import get_tickers_polygon

import numpy as np
from scipy.signal import argrelextrema

from joblib import Parallel, delayed
from gerchik.utils import draw
from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles
from gerchik.utils import check_podzhatie

from alpha.broker import send_stop_order
from alpha.broker import get_account_balance
from bot.bot import bot


deal_size = 1000


async def send_message(msg):
    chat_id = "335442091"
    await bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')


def send_telegram_notification(msg):
    # Start the event loop
    loop = asyncio.get_event_loop()

    # Run the async function
    loop.run_until_complete(send_message(msg))

    # Close the event loop
    loop.close()


def run_me(ticker, progress=0):
    # print(f'  Loading data... {progress:.2f}% done     ', end='\r')

    with open('positions.json', mode='r', encoding='utf-8') as f:
        positions = json.load(f)

    # Make sure we do not send duplicated orders
    if ticker in positions:
        return ticker, 0

    try:
        # take additional 150 days to identify levels properly
        df = get_data(ticker, period='day', days=150, save_data=False)
        df.index = df.index.strftime('%b %d')
    except Exception as e:
        print(e)
        return ticker, -100

    if df is None or df.empty or len(df) < 50:
        return ticker, -100

    df_original = df.copy()
    # df = df.iloc[:-2]  # cut the last day to check

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return ticker, -100

    current_price = df['Close'].tolist()[-1]
    if 1 > current_price or current_price > 100:
        return ticker, -100  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return ticker, -100

    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    open_prices = df['Open'].tolist()
    close_prices = df['Close'].tolist()

    luft = 0.02 * atr

    found_signal = False
    levels = []

    # paranormal bars level search:

    for i, (index, row) in enumerate(df.iterrows()):
        bar_size = row['High'] - row['Low']

        if i < len(df) - 5 and bar_size > 2 * atr:
            level_h = row['High']
            level_l = row['Low']

            for j in range(i+1, len(df) - 1):
                if df['Open'].iloc[j] < level_h < df['Close'].iloc[j]:
                    level_h = 0
                if df['Open'].iloc[j] > level_h > df['Close'].iloc[j]:
                    level_h = 0

                if df['Open'].iloc[j] < level_l < df['Close'].iloc[j]:
                    level_l = 0
                if df['Open'].iloc[j] > level_l > df['Close'].iloc[j]:
                    level_l = 0

            if level_h:
                levels.append({'price': level_h, 'type': 'paranormal bars level'})
            if level_l:
                levels.append({'price': level_h, 'type': 'paranormal bars level'})

    # Check if nearest Highs or Lows form new level:
    if abs(highs[-1] - highs[-2]) < 0.02 or abs(highs[-1] - highs[-3]) < 0.02 or abs(highs[-1] - highs[-4]) < 0.02:
        levels.append({'price': highs[-1], 'type': 'nearest high'})
    if abs(lows[-1] - lows[-2]) < 0.02 or abs(lows[-1] - lows[-3]) < 0.02 or abs(lows[-1] - lows[-4]) < 0.02:
        levels.append({'price': lows[-1], 'type': 'nearest high'})

    # Limit + mirror levels search

    prices = sorted(highs + lows)
    bars_required = 3

    group = []
    previous_price = prices[0]

    for p in prices:
        if 100 * abs(previous_price - p) / p < 0.5 * luft:
            group.append(p)
        else:
            if len(group) >= bars_required:
                level = min(group)

                levels.append({'price': level, 'type': 'mirror level'})

            group = []

        previous_price = p

    # Izlom trenda search

    border = 10

    # Find local minima and maxima indices
    minima_idx = argrelextrema(df['Low'].values, np.less, order=border)[0]
    maxima_idx = argrelextrema(df['High'].values, np.greater, order=border)[0]

    # Get corresponding data points
    local_minima = df.iloc[minima_idx]
    local_maxima = df.iloc[maxima_idx]

    for i, (index, row) in enumerate(local_minima.iterrows()):
        levels.append({'price': row['Low'], 'type': 'izlom trenda'})

    for i, (index, row) in enumerate(local_maxima.iterrows()):
        levels.append({'price': row['High'], 'type': 'izlom trenda'})

    level = check_podzhatie(df)
    if level > 0:
        levels.append({'price': level, 'type': 'podzhatie'})

    # Choosing the right level:

    selected_level = 0
    for level in levels:
        # Check if level is clear:

        k = 0
        for i in range(0, len(df)):
            if lows[-i] < level['price'] < highs[-i]:
                k += 1

            # This is for short orders with LP
            # if lows[-i] > level['price']:
            #     k += 1

            # This is for long orders with LP
            if highs[-i] < level['price']:
                k += 1

        if k < 2:
            if lows[-1] < level['price'] < open_prices[-1] and close_prices[-1] < level['price']:
                found_signal = True
                selected_level = level

    if found_signal:
        buy_price = round(selected_level['price'] + luft, 2)
        stop_loss = round(buy_price - 0.3 * atr, 2)
        take_profit = round(buy_price + 10 * abs(buy_price - stop_loss), 2)   # 3 takes - one on x7, one on x10 one on x20

        # If we didn't spend enough fuel (ATR) or we spent too much - do not trade this
        previous_open = df['Open'].values[-1]
        proshli_do_urovnia = 100 * abs(selected_level['price'] - previous_open) / atr
        if proshli_do_urovnia < 50 or proshli_do_urovnia > 300:
            return ticker, 0
        ####

        """
        boxes = []
        levels_to_draw = []

        draw(
            df_original.iloc[-40:].copy(), file_name=f'{ticker}', ticker=ticker,
            level=selected_level['price'], boxes=boxes, second_levels=levels_to_draw, future=0,
            buy_price=buy_price, stop_loss=stop_loss, take_profit=take_profit, buy_index=df.index.values[-1],
            zig_zag=False,
        )
        """

        # TODO check here if we have open position for this ticker already
        try:
            quantity = round(deal_size / buy_price)
            delta_price = round(buy_price + 0.1 * atr, 2)  # delta price helps to create Stop Limit order
            send_stop_order(
                ticker, 'BUY', quantity, price=buy_price, delta_price=delta_price,
                stop_loss=stop_loss, take_profit=take_profit
            )
        except Exception as e:
            print(f'Failed to send an order: {e}')
            send_telegram_notification(f'Failed to send an order: {e}')

        account_balance = get_account_balance()

        try:
            msg = f'Buy order *{ticker}* for ${buy_price:.2f} and stop loss: ${stop_loss:.2f}\n'
            msg += f'Stop loss {100 * abs(stop_loss-buy_price) / buy_price:.2f}%\n'
            msg += f'Take profit {100 * abs(take_profit - buy_price) / buy_price:.2f}%\n'
            msg += f'Account Balance: {account_balance}'
            send_telegram_notification(msg)
        except:
            pass

        # TODO - learn how to extract tickers from broker data, do not save it locally
        with open('positions.json', mode='w+', encoding='utf-8') as f:
            positions.append(ticker)
            json.dump(positions, f)

    return ticker, 0


if __name__ == "__main__":
    print(f'{datetime.datetime.now().strftime("%b %d %Y %H:%M:%S")}')

    # Do not run this script on weekends (Monday == 0)
    if datetime.datetime.now().weekday() >= 5:
        print('Non working day, skipping')
        exit(0)

    # Only look for opportunities from 15:00 to 21:00 by Berlin
    if 15 <= datetime.datetime.now().hour <= 21:

        print('Starting threads...')
        TICKERS = get_tickers_polygon(limit=5000)  # 2000
        total_results = []

        with open('bad_tickers.json', encoding='utf=8', mode='r') as f:
            bad_tickers = json.load(f)

        # we filter bad tickers beforehand to not waste time in the future
        TICKERS = [t for t in TICKERS if t not in bad_tickers]

        result = Parallel(n_jobs=-1, max_nbytes='200M', backend="multiprocessing", timeout=100)(
            delayed(run_me)(ticker, 100*i/len(TICKERS)) for i, ticker in enumerate(TICKERS)
        )

        for r in result:
            if r[1] == -100:
                bad_tickers.append(r[0])

        with open('bad_tickers.json', encoding='utf=8', mode='w+') as f:
            json.dump(bad_tickers, f)

        print()
    else:
        print('Outside of the working hours, skipping')
