# Failed: 52, passed: 96, win rate: 64.86%
# Profit: 417.21%
#

import warnings
warnings.filterwarnings("ignore")

from crypto_forex.utils import ALL_TICKERS

from tqdm import tqdm
from joblib import Parallel, delayed

import json

from app.utils import get_data_alpha
from bot.utils import get_tickers_polygon
from bot.utils import get_data

from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles


TICKERS = get_tickers_polygon(limit=5000)  # this is for shares
# TICKERS = [t[2:] for t in sorted(ALL_TICKERS)]  # this is for forex
ALL_OPERATIONS = []
max_days = 1000  # collect data for about 5 years

TICKERS = TICKERS[100:200]


def run_me(ticker) -> list:
    WW = []
    result = []

    # df = get_data_alpha(ticker, interval='Daily', limit=max_days)
    df = get_data(ticker, period='day', days=max_days)

    # make sure it is mature stock and we will be able to calculate EMA200
    if df is None or len(df) < 201:
        return []

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 100:  # only take shares with 100k+ average volume
        return []

    current_price = df['Close'].tolist()[-1]
    if current_price < 1:  # this actually helps because penny stocks behave differently
        return []  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return []

    # first - calculate EMAs so we have this data
    for s in range(5, 75, 3):
        for b in range(13, 200, 7):
            if s == b:
                continue

            if f'EMA{s}' not in df:
                df.ta.ema(length=s, append=True, col_names=(f'EMA{s}',))
            if f'EMA{b}' not in df:
                df.ta.ema(length=b, append=True, col_names=(f'EMA{b}',))

    # Cut on history and evaluation dataset:
    df_last_month = df.iloc[-45:].copy()  # we need to include 5 days of history, but we will not make deals
                                          # for the first 5 days and last 20 days of this data frame
    df = df.iloc[:-40].copy()  # cut here the last month (we trade first 20 days, and also waiting for all results 20 days more


    for s in range(5, 75, 3):
        for b in range(13, 200, 7):
            if s == b:
                continue

            RESULTS = []

            prev_s = 0
            prev_b = 0
            for i, (index, row) in enumerate(df.iterrows()):
                if max(s, b) < i < len(df) - 20:
                    if row[f'EMA{s}'] > row[f'EMA{b}'] and prev_s < prev_b:
                        volatility = max(df['High'].values[i-5:i+1]) - min(df['Low'].values[i-5:i+1])

                        buy_price = df['Open'].values[i + 1]
                        stop_loss = buy_price - volatility / 2
                        take_profit = buy_price + 4 * abs(stop_loss - buy_price)
                        profit = 0

                        for j in range(i + 1, len(df)):
                            if profit == 0:
                                if df['Low'].values[j] < stop_loss:
                                    profit = -abs(buy_price - stop_loss) / buy_price
                                elif df['High'].values[j] > take_profit:
                                    profit = abs(take_profit - buy_price) / buy_price

                        if profit != 0 and abs(stop_loss - buy_price) / buy_price < 0.03:
                            RESULTS.append(profit)

                    prev_s = row[f'EMA{s}']
                    prev_b = row[f'EMA{b}']

            failed = sum([1 for r in RESULTS if r < 0])
            passed = sum([1 for r in RESULTS if r > 0])

            # if we had some good cases in the history and win rate is good,
            # we hope it will also work for us in the future (but no guarantee)
            if len(RESULTS) > 5 and passed / (passed + failed) > 0.5:
                WW.append({'combo': [s, b], 'win rate': 100 * passed / (passed + failed)})

    # here we only take best combos for each ticker, and we do not trading with combos
    # that do not have traces of good results in the past
    for i, (index, row) in enumerate(df_last_month.iterrows()):
        if len(df_last_month) - 20 > i > 5:
            perform = 0

            for combo in WW:
                s, b = combo['combo']

                if row[f'EMA{s}'] > row[f'EMA{b}'] and df_last_month[f'EMA{s}'].values[i-1] < df_last_month[f'EMA{b}'].values[i-1]:
                    volatility = max(df_last_month['High'].values[i - 5:i + 1]) - min(df_last_month['Low'].values[i - 5:i + 1])

                    buy_price = df_last_month['Open'].values[i + 1]
                    stop_loss = buy_price - volatility / 2

                    if abs(stop_loss - buy_price) / buy_price < 0.03:
                        perform = 1

            if perform:
                volatility = max(df_last_month['High'].values[i - 5:i + 1]) - min(
                    df_last_month['Low'].values[i - 5:i + 1])
                buy_price = df_last_month['Open'].values[i + 1]
                stop_loss = buy_price - volatility / 2
                take_profit = buy_price + 4 * abs(stop_loss - buy_price)
                profit = 0

                for j in range(i + 1, len(df_last_month)):
                    if profit == 0:
                        if df_last_month['Low'].values[j] < stop_loss:
                            profit = -abs(buy_price - stop_loss) / buy_price
                        elif df_last_month['High'].values[j] > take_profit:
                            profit = abs(take_profit - buy_price) / buy_price

                if profit != 0:
                    result.append(profit)

    print(result)
    return result


if __name__ == "__main__":
    print('Preparing training dataset...')
    results = Parallel(n_jobs=-1, backend="multiprocessing", timeout=300)(
        delayed(run_me)(ticker) for ticker in TICKERS
    )
    print(results)
    for r in results:
        if r:
            ALL_OPERATIONS += r

    failed = sum([1 for r in ALL_OPERATIONS if r < 0])
    passed = sum([1 for r in ALL_OPERATIONS if r > 0])
    profit_total = sum(ALL_OPERATIONS)

    print(f'Failed: {failed}, passed: {passed}, win rate: {100 * passed / (passed+failed):.2f}%')
    print(f'Profit: {100 * profit_total:.2f}%')
