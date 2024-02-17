# Failed: 52, passed: 96, win rate: 64.86%
# Profit: 417.21%
#

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


def run_me(ticker, progress_value) -> list:
    WW = []
    result = []

    # df = get_data_alpha(ticker, interval='Daily', limit=max_days)
    df = get_data(ticker, period='hour', days=max_days)

    # make sure it is mature stock and we will be able to calculate EMA200
    if df is None or len(df) < 360:
        return []

    current_price = df['Close'].tolist()[-1]
    if current_price < 1:  # this actually helps because penny stocks behave differently
        return []  # ignore penny stocks and huge stocks

    params = []
    # first - calculate EMAs so we have this data
    file_name = f'collect_data/calculated/h_{ticker}.parquet'
    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)
    else:
        for s in range(21, 200, 7):

            if f'EMA{s}' not in df:
                df.ta.ema(length=s, append=True, col_names=(f'EMA{s}',))
            if f'SMA{s}' not in df:
                df.ta.sma(length=s, append=True, col_names=(f'SMA{s}',))

            """
            if f'fwma{s}' not in df:
                df.ta.fwma(length=s, append=True, col_names=(f'fwma{s}',))
            if f'tema{s}' not in df:
                df.ta.tema(length=s, append=True, col_names=(f'tema{s}',))
            if f't3_{s}' not in df:
                df.ta.t3(length=s, append=True, col_names=(f't3_{s}',))
            if f'zlma{s}' not in df:
                df.ta.zlma(length=s, append=True, col_names=(f'zlma{s}',))
            """

            # Precalculate "previous" values to confirm crossing using one row
            df[f'EMA{s}_prev'] = df[f'EMA{s}'].shift(1)
            df[f'SMA{s}_prev'] = df[f'SMA{s}'].shift(1)
            """
            df[f'fwma{s}_prev'] = df[f'fwma{s}'].shift(1)
            df[f'tema{s}_prev'] = df[f'tema{s}'].shift(1)
            df[f't3_{s}_prev'] = df[f't3_{s}'].shift(1)
            df[f'zlma{s}_prev'] = df[f'zlma{s}'].shift(1)
            """

            params += [f'EMA{s}', f'SMA{s}']  # , f'fwma{s}', f'tema{s}', f't3_{s}', f'zlma{s}']

        def calc_volatility(highs, lows):
            return highs.max() - lows.min()

        df['Volatility'] = rolling_apply(calc_volatility, 5, df['High'].values, df['Low'].values)

        df.to_parquet(file_name)

    # Cut on history and evaluation dataset:
    df_last_month = df.iloc[-120:]  # we need to include 20 hours of history, but we will not make deals
                                    # for the first 20 hours and last 5 hours of this data frame
    df = df.iloc[:-160]  # cut here the last month (we trade first 20 days, and also waiting for all results 20 days more

    file_name = f'collect_data/calculated/WW_hour_{ticker}.json'
    if os.path.isfile(file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            WW = json.load(f)
    else:
        for s in params:
            for b in params:
                if s == b:
                    continue

                RESULTS = []

                for i, (index, row) in enumerate(df.iterrows()):
                    if 200 < i < len(df) - 20:
                        if row[s] > row[b] and row[f'{s}_prev'] < row[f'{b}_prev']:
                            buy_price = df['Open'].values[i + 1]
                            stop_loss = buy_price - row['Volatility'] / 2
                            take_profit = buy_price + 5 * abs(stop_loss - buy_price)
                            profit = 0

                            if abs(stop_loss - buy_price) / buy_price < 0.03:
                                for j in range(i + 1, len(df)):
                                    if profit == 0:
                                        if df['Low'].values[j] < stop_loss:
                                            profit = -abs(buy_price - stop_loss) / buy_price
                                        elif df['High'].values[j] > take_profit:
                                            profit = abs(take_profit - buy_price) / buy_price

                                if profit != 0:
                                    RESULTS.append(profit)

                failed = sum([1 for r in RESULTS if r < 0])
                passed = sum([1 for r in RESULTS if r > 0])

                # if we had some good cases in the history and win rate is good,
                # we hope it will also work for us in the future (but no guarantee)
                if len(RESULTS) > 2:
                    win_rate = 100 * passed / (passed + failed)
                    if win_rate > 60:
                        WW.append({'combo': [s, b], 'win rate': win_rate})

        with open(file_name, encoding='utf-8', mode='w+') as f:
            json.dump(WW, f)

    # here we only take best combos for each ticker, and we do not trading with combos
    # that do not have traces of good results in the past

    for i, (index, row) in enumerate(df_last_month.iterrows()):
        if len(df_last_month) - 20 > i > 5:
            perform = 0

            for combo in WW:
                s, b = combo['combo']

                if row[s] > row[b] and row[f'{s}_prev'] < row[f'{b}_prev']:
                    buy_price = df_last_month['Open'].values[i + 1]
                    stop_loss = buy_price - row['Volatility'] / 2

                    # make sure the stop is not more than 3%
                    if abs(stop_loss - buy_price) / buy_price < 0.03:
                        # check that this strategy bring good results for the current period
                        if len(result) < 2 or sum(result) > 0:
                            if row['Close'] > row['Open']:   # this helps to increase win rate
                                perform = 1

            if perform:
                # day_of_week = index.weekday()

                buy_price = df_last_month['Open'].values[i + 1]
                stop_loss = buy_price - row['Volatility'] / 2
                take_profit = buy_price + 3 * abs(stop_loss - buy_price)
                profit = 0

                for j in range(i + 1, len(df_last_month)):
                    if profit == 0:
                        if df_last_month['Low'].values[j] < stop_loss:
                            profit = -abs(buy_price - stop_loss) / buy_price
                        elif df_last_month['High'].values[j] > take_profit:
                            profit = abs(take_profit - buy_price) / buy_price

                if profit != 0:
                    result.append(profit)

                    """
                    draw(
                        df_last_month.copy(), file_name=ticker + f' {i}', ticker=ticker,
                        level=0, boxes=[], stop_loss=stop_loss, take_profit=take_profit,
                        buy_price=buy_price, buy_index=i+1
                    )
                    """

                    # print(f'  {progress_value:.2f}% done...', end='\r')
                    # return result

    print(f'  {progress_value:.2f}% done...', end='\r')
    return result


if __name__ == "__main__":
    print('Preparing training dataset...')

    TICKERS = get_tickers_polygon(limit=5000)  # this is for shares
    TICKERS = TICKERS[:1000]

    results = Parallel(n_jobs=-1, backend="multiprocessing", timeout=500)(
        delayed(run_me)(ticker, 100*e/len(TICKERS)) for e, ticker in enumerate(TICKERS)
    )

    for r in results:
        if r:
            ALL_OPERATIONS += r

    failed = sum([1 for r in ALL_OPERATIONS if r < 0])
    passed = sum([1 for r in ALL_OPERATIONS if r > 0])
    profit_total = sum(ALL_OPERATIONS)

    print(f'Failed: {failed}, passed: {passed}, win rate: {100 * passed / (passed+failed):.2f}%')
    print(f'Profit: {100 * profit_total:.2f}%')
