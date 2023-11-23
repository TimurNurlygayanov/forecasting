
from bot.utils import get_data
from bot.utils import get_tickers_polygon

from tqdm import tqdm
from scipy.signal import argrelextrema
import numpy as np

from joblib import Parallel, delayed


TICKERS = get_tickers_polygon(limit=5000)  # 2000
RESULTS = []


def check_for_bad_candles(ticker, df):
    gaps = 0

    for i, (index, row) in enumerate(df.iterrows()):
        if i > 0:
            if abs(row['High'] - row['Low']) < 0.4 * row['ATR']:
                gaps += 1
            if abs(row['High'] - row['Low']) < 0.1:
                gaps += 1

    if gaps / len(df) > 0.05:
        return True

    return False


def run_me(ticker):
    global RESULTS

    df = get_data(ticker, period='day')

    if df is None or df.empty or len(df) < 20:
        return None

    df['ATR'] = 0
    df.ta.atr(append=True, col_names=('ATR',))
    if check_for_bad_candles(ticker, df):
        return None

    current_price = df['Close'].tolist()[-1]

    if 3 > current_price or current_price > 300:
        return None  # ignore penny stocks and huge stocks

    # price_diff = 0.1 * df['ATR'].tolist()[-1]

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000

    if average_volume < 1000:
        return None

    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    open_prices = df['Open'].tolist()
    close_prices = df['Close'].tolist()
    threshold = 0.02

    found_signal = False
    if abs(lows[-1] - lows[-2]) < threshold:
        if close_prices[-1] < open_prices[-1]:

            if abs(close_prices[-1] - lows[-1]) < abs(close_prices[-1] - highs[-1]):
                found_signal = True

                # print(ticker, open_prices[-1], close_prices[-1], lows[-1])

    if abs(highs[-1] - highs[-2]) < threshold:
        if close_prices[-1] > open_prices[-1]:

            if abs(close_prices[-1] - highs[-1]) < abs(close_prices[-1] - lows[-1]):
                found_signal = True

                # print(ticker, open_prices[-1], close_prices[-1], highs[-1])

    if found_signal:
        RESULTS.append(ticker)
        print(len(RESULTS), RESULTS)


Parallel(n_jobs=-1, require='sharedmem', timeout=20)(delayed(run_me)(ticker) for ticker in TICKERS)

for t in RESULTS:
    print(t)

print(len(RESULTS))
