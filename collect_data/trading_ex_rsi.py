# based on this idea https://www.youtube.com/watch?v=3zI_l_P-lF8
#
import random
import warnings
warnings.filterwarnings("ignore")

# from crypto_forex.utils import ALL_TICKERS
# from app.utils import get_data_alpha

from datetime import datetime

import json
from joblib import Parallel, delayed

import os
import pandas as pd
from numpy_ext import rolling_apply

from bot.utils import get_tickers_polygon
from bot.utils import get_data

from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles
from gerchik.utils import draw


def calculate_volatility(high_window, low_window):
    return high_window.max() - low_window.min()


# TICKERS = [t[2:] for t in sorted(ALL_TICKERS)]  # this is for forex
ALL_OPERATIONS = []
max_days = 200  # collect data for about 5 years


def run_me(ticker, day=0, progress_value=0) -> tuple:
    # first - calculate EMAs so we have this data
    file_name = f'collect_data/calculated/h_{ticker}.parquet'
    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)

        df.ta.ema(length=7, append=True, col_names=(f'EMA3',))  # we take this instead of price itself
        df['EMA3_prev'] = df['EMA3'].shift(1)

        for e in range(50, 150, 10):
            df.ta.ema(length=e, append=True, col_names=(f'EMA{e}',))  # we take this instead of price itself
            df[f'EMA{e}_prev'] = df[f'EMA{e}'].shift(1)

        # Cut on history and evaluation dataset:
        df = df.iloc[-max_days - 1:].copy()  # we need to include 20 hours of history, but we will not make deals
                                    # for the first 20 hours and last 5 hours of this data frame

        perform = 0
        row = df.iloc[day]  # we only checking the data from previous day
        row_prev = df.iloc[day-1]

        # if row['EMA21'] > row['EMA50'] and row_prev['EMA21'] < row_prev['EMA50']:
        stop_loss = 0
        for e in range(50, 150, 10):
            if row['EMA3'] < row[f'EMA{e}'] and row['EMA3_prev'] > row[f'EMA{e}_prev']:
                perform = 1
                stop_loss = row[f'EMA{e}'] - row['ATR']

        if perform:
            buy_price = df['Open'].values[day + 1]
            take_profit = buy_price + 5 * abs(stop_loss - buy_price)

            return buy_price, stop_loss, take_profit

    return 0, 0, 0


def check_profit(ticker, hour, buy_price, stop_loss, take_profit):
    profit = 0

    file_name = f'collect_data/calculated/h_{ticker}.parquet'
    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)

        # Cut on history and evaluation dataset:
        df = df.iloc[-max_days-1:].copy()  # we need to include 20 hours of history, but we will not make deals
                                           # for the first 20 hours and last 5 hours of this data frame

        if df['Low'].values[hour] < stop_loss:
            profit = -abs(buy_price - stop_loss) / buy_price
        elif df['High'].values[hour] > take_profit:
            profit = abs(take_profit - buy_price) / buy_price

    return profit


def quick_sell(ticker, chunk, i=0):
    file_name = f'collect_data/calculated/h_{ticker}.parquet'
    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)
        if i == 0:
            price = df['Close'].tolist()[-1]
        else:
            price = df['Close'].tolist()[i]

        return price * chunk


if __name__ == "__main__":
    print('Preparing training dataset...')

    TICKERS = get_tickers_polygon(limit=5000)  # this is for shares
    # TICKERS = TICKERS[:1000]

    current_deals = []
    current_free_money = 10000
    total_results = []

    for i in range(0, max_days):
        hour = i - max_days  # we starting from 100 hours ago
        current_deals_copy = current_deals.copy()
        for deal in current_deals_copy:
            profit = check_profit(deal['t'], hour, deal['buy_price'], deal['stop_loss'], deal['take_profit'])

            if profit:
                current_deals.remove(deal)

                if profit > 0:
                    current_free_money += deal['take_profit'] * deal['c']
                else:
                    current_free_money += deal['stop_loss'] * deal['c']

                total_results.append(profit)

                print(f'Sell {deal["t"]} with profit {100*profit:.2f}%')

        random.shuffle(TICKERS)
        for t in TICKERS:
            buy_price, stop_loss, take_profit = run_me(ticker=t, day=hour, progress_value=i)

            if buy_price > 0:
                n = int(700/buy_price)

                if n > 1 and current_free_money > n * buy_price:  # buy only if we can buy at least 1 share
                        total_price = n * buy_price
                        current_free_money -= total_price
                        current_deals.append({
                            't': t, 'c': n, 'buy_price': buy_price,
                            'stop_loss': stop_loss, 'take_profit': take_profit
                        })

                        print(f'Buy {t} for total ${buy_price*n:.2f}')


        money = current_free_money
        for d in current_deals:
            # do not sell but pretend we sell right away now
            money += quick_sell(ticker=d['t'], chunk=d['c'], i=hour)

        print(f'  Step {i} current money: {money:.2f}, results: {len(total_results)}')

    # sell everything in the end:
    for deal in current_deals:
        last_sell = quick_sell(ticker=deal['t'], chunk=deal['c'])
        print(f'After sale {deal["t"]} for total {last_sell:.2f}')
        current_free_money += last_sell

    print('* ' * 20)
    print(f'current money: {current_free_money:.2f}, results: {len(total_results)}')
    print(f'Win Rate: {100 * sum([1 for r in total_results if r > 0]) / len(total_results):.2f} ')
