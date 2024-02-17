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
import numpy as np
import pandas as pd
from numpy_ext import rolling_apply

from bot.utils import get_tickers_polygon
from bot.utils import get_data


def calculate_volatility(high_window, low_window):
    return high_window.max() - low_window.min()


ALL_OPERATIONS = []
max_days = 200  # collect data for about 5 years
result_df = None


def run_me(ticker) -> list:
    global result_df

    WW = []
    result = []

    # df = get_data_alpha(ticker, interval='Daily', limit=max_days)
    df = get_data(ticker, period='hour', days=max_days)

    # make sure it is mature stock and we will be able to calculate EMA200
    if df is None or len(df) < 360:
        return []

    params = []
    # first - calculate EMAs so we have this data
    file_name = f'collect_data/calculated/h_{ticker}.parquet'
    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)
    else:
        for s in range(5, 50, 3):

            if f'EMA{s}' not in df:
                df.ta.ema(length=s, append=True, col_names=(f'EMA{s}',))
            if f'SMA{s}' not in df:
                df.ta.sma(length=s, append=True, col_names=(f'SMA{s}',))
            if f'fwma{s}' not in df:
                df.ta.fwma(length=s, append=True, col_names=(f'fwma{s}',))
            if f'tema{s}' not in df:
                df.ta.tema(length=s, append=True, col_names=(f'tema{s}',))
            if f'wma{s}' not in df:
                df.ta.wma(length=s, append=True, col_names=(f'wma{s}',))
            if f'zlma{s}' not in df:
                df.ta.zlma(length=s, append=True, col_names=(f'zlma{s}',))

            # Precalculate "previous" values to confirm crossing using one row
            df[f'EMA{s}_prev'] = df[f'EMA{s}'].shift(1)
            df[f'SMA{s}_prev'] = df[f'SMA{s}'].shift(1)
            df[f'fwma{s}_prev'] = df[f'fwma{s}'].shift(1)
            df[f'tema{s}_prev'] = df[f'tema{s}'].shift(1)
            df[f'wma{s}_prev'] = df[f'wma{s}'].shift(1)
            df[f'zlma{s}_prev'] = df[f'zlma{s}'].shift(1)

            params += [f'EMA{s}', f'SMA{s}', f'fwma{s}', f'tema{s}', f'wma{s}', f'zlma{s}']

        def calc_volatility(highs, lows):
            return highs.max() - lows.min()

        df['Volatility'] = rolling_apply(calc_volatility, 5, df['High'].values, df['Low'].values)

        df.to_parquet(file_name)

    file_name = f'collect_data/calculated/WW_hour_{ticker}.json'
    if os.path.isfile(file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            WW = json.load(f)

    # here we only take best combos for each ticker, and we do not trading with combos
    # that do not have traces of good results in the past

    df = df.iloc[:-160].copy()  # make sure we cut last 100 hours from the data source, so we can verify model later

    if WW:
        df['ticker'] = ticker
        df['result'] = '-'
        for i, (index, row) in enumerate(df.iterrows()):
            if len(df) - 5 > i > 5:
                perform = 0

                for combo in WW:
                    s, b = combo['combo']

                    if not perform:
                        if row[s] > row[b] and row[f'{s}_prev'] < row[f'{b}_prev']:
                            buy_price = df['Open'].values[i + 1]
                            stop_loss = buy_price - row['Volatility'] / 2

                            # make sure the stop is not more than 3%
                            if abs(stop_loss - buy_price) / buy_price < 0.03:
                                # check that this strategy bring good results for the current period
                                if len(result) < 2 or sum(result) > 0:
                                    if row['Close'] > row['Open']:   # this helps to increase win rate
                                        perform = 1

                if perform:
                    buy_price = df['Open'].values[i + 1]
                    stop_loss = buy_price - row['Volatility'] / 2
                    take_profit = buy_price + 4 * abs(stop_loss - buy_price)
                    profit = 0

                    for j in range(i + 1, len(df)):
                        if profit == 0:
                            if df['Low'].values[j] < stop_loss:
                                profit = -abs(buy_price - stop_loss) / buy_price
                            elif df['High'].values[j] > take_profit:
                                profit = abs(take_profit - buy_price) / buy_price

                    if profit != 0:
                        result.append(profit)
                        df.loc[index, 'result'] = "good" if profit > 0 else "bad"

        df.set_index('ticker', append=True, inplace=True)

        if result_df is None:
            result_df = df.copy()
        else:
            result_df = pd.concat([result_df, df])

            result_df.to_parquet('collect_data/data.parquet')

    return result


if __name__ == "__main__":
    print('Preparing training dataset...')

    TICKERS = get_tickers_polygon(limit=5000)  # this is for shares
    TICKERS = TICKERS[:2000]

    for ticker in TICKERS:
        run_me(ticker)

    bad = (result_df['result'] == "bad").sum()
    good = (result_df['result'] == "good").sum()

    print(f"Good results: {good}, bad results: {bad}, success rate {100 * good / (good + bad):.2f}")

    train_X = []
    train_y = []
    for i, (index, row) in enumerate(result_df.iterrows()):
        if row['result'] in ['good', 'bad']:
            r = []
            for p in result_df.columns.tolist():
                if p != 'result':
                    value = 0 if np.isnan(row[p]) else row[p]
                    r.append(value)

            train_X.append(r)
            train_y.append(1 if row['result'] == 'good' else 0)
