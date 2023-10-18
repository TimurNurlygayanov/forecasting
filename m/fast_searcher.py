
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic

from tqdm import tqdm

from joblib import Parallel, delayed

from bot.utils import get_data
from bot.utils import get_tickers_polygon


TICKERS = get_tickers_polygon(limit=5000)
RESULTS = {}
SUPER_TREND_CHANGED = {}
HIGH_VOLATILITY = {}


def check(ticker, df):
    global HIGH_VOLATILITY
    global RESULTS

    last_index = df.shape[0] - 1

    if last_index < 100:
        return None

    # df.ta.atr(append=True, col_names=('ATR',))
    # df.ta.ema(length=200, append=True, col_names=('EMA200',))
    df.ta.supertrend(append=True, length=10, multiplier=3.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    # df.ta.bbands(close='Close', length=20, std=2, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

    dfHA = df.ta.ha()
    dfHA.rename(columns={'HA_open': 'Open', 'HA_close': 'Close', 'HA_low': 'Low', 'HA_high': 'High'}, inplace=True)
    # dfHA.ta.supertrend(append=True, length=34, multiplier=3.0,
    #                    col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    last_row_ha = dfHA.iloc[-1]
    prev_row_ha = dfHA.iloc[-2]

    volatility = (max(df['High'].values[-10:]) - min(df['Low'].values[-10:])) / last_row['Close']
    average_volume = max(df['volume'].values[-10:]) / 10

    if 3 < last_row['Close'] < 400:
        if volatility > 0.01 and average_volume > 1000:
            HIGH_VOLATILITY[ticker] = volatility

            if prev_row_ha['Close'] < prev_row_ha['Open']:
                if last_row_ha['Open'] < last_row_ha['Close']:
                    print('> ', ticker)
                    RESULTS[ticker] = sum(df['volume'].values[-10:]) / 10

            d = prev_row_ha['High'] - prev_row_ha['Low']
            b = abs(prev_row_ha['Close'] - prev_row_ha['Open'])

            if prev_row['S_trend_d'] != last_row['S_trend_d']:
                SUPER_TREND_CHANGED[ticker] = sum(df['volume'].values[-10:]) / 10
                print('s ', ticker)


def run_me(ticker):
    df = get_data(ticker, period='hour', multiplier=1, save_data=False)
    check(ticker, df)


if __name__ == '__main__':

    TICKERS = TICKERS[:4000]
    Parallel(n_jobs=-1, require='sharedmem')(delayed(run_me)(element) for element in TICKERS)

    print('Most volatile shares:')
    for ticker, volatility in sorted(HIGH_VOLATILITY.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(ticker, volatility)

    if len(RESULTS) > 0:
        print('Found good candidates:')
        # medium_value = sum([v for v in RESULTS.values()]) / len(RESULTS)
        for ticker in RESULTS:
            # if RESULTS[ticker] > medium_value:
            print(ticker)
    else:
        print('No good candidates so far...')

