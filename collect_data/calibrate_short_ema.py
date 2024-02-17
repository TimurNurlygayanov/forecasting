
import warnings
warnings.filterwarnings("ignore")

# from crypto_forex.utils import ALL_TICKERS
# from app.utils import get_data_alpha

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
max_days = 100  # collect data for about 5 years


def run_me(ticker, progress_value) -> list:
    result = []

    # df = get_data_alpha(ticker, interval='Daily', limit=max_days)
    df = get_data(ticker, period='hour', days=max_days)

    # make sure it is mature stock and we will be able to calculate EMA200
    if df is None or len(df) < 201:
        return []

    # average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    # if average_volume < 100:  # only take shares with 100k+ average volume
    #     return []

    current_price = df['Close'].tolist()[-1]
    if current_price < 1:  # this actually helps because penny stocks behave differently
        return []  # ignore penny stocks and huge stocks

    # atr = calculate_atr(df)
    # if check_for_bad_candles(df, atr):
    #     return []

    def calc_volatility(highs, lows):
        return highs.max() - lows.min()

    df['Volatility'] = rolling_apply(calc_volatility, 5, df['High'].values, df['Low'].values)

    df.ta.ema(length=7, append=True, col_names=('EMA_short',))
    df.ta.ema(length=50, append=True, col_names=('EMA_long',))

    df['EMA_short_prev'] = df['EMA_short'].shift(1)
    df['EMA_long_prev'] = df['EMA_long'].shift(1)

    # Cut on history and evaluation dataset:
    df_last_month = df.iloc[-165:]  # we need to include 5 days of history, but we will not make deals
                                   # for the first 5 days and last 20 days of this data frame

    for i, (index, row) in enumerate(df_last_month.iterrows()):
        if len(df_last_month) - 5 > i > 50:
            if row['EMA_short'] > row['EMA_long'] and row['EMA_short_prev'] < row['EMA_long_prev']:
                buy_price = df_last_month['Open'].values[i + 1]
                stop_loss = buy_price - row['Volatility'] / 2   #  min(df_last_month['Low'].values[i-3:i + 1]) - 0.05
                take_profit = buy_price + 4 * abs(stop_loss - buy_price)
                profit = 0

                for j in range(i + 1, len(df_last_month)):
                    if profit == 0:
                        if df_last_month['Low'].values[j] < stop_loss:
                            profit = -abs(buy_price - stop_loss) / buy_price
                        elif df_last_month['High'].values[j] > take_profit:
                            profit = abs(take_profit - buy_price) / buy_price

                if profit != 0 and abs(stop_loss - buy_price) / buy_price < 0.03:
                    result.append(profit)

                    # print(f'  {progress_value:.2f}% done...', end='\r')
                    # return result

    print(f'  {progress_value:.2f}% done...', end='\r')
    return result


if __name__ == "__main__":
    print('Preparing training dataset...')

    TICKERS = get_tickers_polygon(limit=5000)  # this is for shares
    # TICKERS = TICKERS[:200]

    results = Parallel(n_jobs=-1, backend="multiprocessing", timeout=300)(
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
