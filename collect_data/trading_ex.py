

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

        # Cut on history and evaluation dataset:
        df = df.iloc[-100:].copy()  # we need to include 20 hours of history, but we will not make deals
                                    # for the first 20 hours and last 5 hours of this data frame

        file_name = f'collect_data/calculated/WW_hour_{ticker}.json'
        if os.path.isfile(file_name):
            with open(file_name, encoding='utf-8', mode='r') as f:
                WW = json.load(f)

            perform = 0
            row = df.iloc[day]  # we only checking the data from previous day
            for combo in WW:
                s, b = combo['combo']

                if row[s] > row[b] and row[f'{s}_prev'] < row[f'{b}_prev']:
                    buy_price = df['Open'].values[day + 1]  # day + 1 is today - the open price
                    stop_loss = buy_price - row['Volatility'] / 2

                    # make sure the stop is not more than 3%
                    if abs(stop_loss - buy_price) / buy_price < 0.03:
                        if row['Close'] > row['Open']:   # this helps to increase win rate
                            perform = 1

            if perform:
                # day_of_week = index.weekday()

                buy_price = df['Open'].values[day + 1]
                stop_loss = buy_price - row['Volatility'] / 2
                take_profit = buy_price + 3 * abs(stop_loss - buy_price)

                return buy_price, stop_loss, take_profit

    # print(f'  {progress_value:.2f}% done...', end='\r')
    return 0, 0, 0


def check_profit(ticker, hour, buy_price, stop_loss, take_profit):
    profit = 0

    file_name = f'collect_data/calculated/h_{ticker}.parquet'
    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)

        # Cut on history and evaluation dataset:
        df = df.iloc[-100:].copy()  # we need to include 20 hours of history, but we will not make deals
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
    TICKERS = TICKERS[:1000]

    current_deals = []
    current_free_money = 10000
    total_results = []

    for i in range(0, 100):

        current_deals_copy = current_deals.copy()
        for deal in current_deals_copy:
            profit = check_profit(deal['t'], i, deal['buy_price'], deal['stop_loss'], deal['take_profit'])

            if profit:
                current_deals.remove(deal)

                if profit > 0:
                    current_free_money += deal['take_profit'] * deal['c']
                else:
                    current_free_money += deal['stop_loss'] * deal['c']

                total_results.append(profit)

                print(f'Sell {deal["t"]} with profit {100*profit:.2f}%')

        for t in TICKERS:
            hour = i - 100  # we starting from 100 hours ago
            buy_price, stop_loss, take_profit = run_me(ticker=t, day=hour, progress_value=i)

            if buy_price > 0:
                if current_free_money > buy_price * 3:
                    n = int(10 / (100 * abs(buy_price - stop_loss) / buy_price))

                    if n > 1 and current_free_money > n * buy_price:  # buy only if we can buy at least 1 share
                        if n * buy_price > 200:
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
            money += quick_sell(ticker=d['t'], chunk=d['c'], i=i-100)

        print(f'  Step {i} current money: {money:.2f}, results: {len(total_results)}')

    # sell everything in the end:
    for deal in current_deals:
        last_sell = quick_sell(ticker=deal['t'], chunk=deal['c'])
        print(f'After sale {deal["t"]} for total {last_sell:.2f}')
        current_free_money += last_sell

    print('* ' * 20)
    print(f'current money: {current_free_money:.2f}, results: {len(total_results)}')
    print(f'Win Rate: {100 * sum([1 for r in total_results if r > 0]) / len(total_results):.2f} ')
