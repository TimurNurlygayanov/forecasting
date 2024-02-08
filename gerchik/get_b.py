from bot.utils import get_data
from bot.utils import get_tickers_polygon

import numpy as np
from scipy.signal import argrelextrema

from joblib import Parallel, delayed
from utils import draw

from utils import calculate_atr
from utils import check_for_bad_candles


TICKERS = get_tickers_polygon(limit=5000)  # 2000
MODE = 'test2'


def run_me(ticker):

    df_original = get_data(ticker, period='day', days=250, save_data=False)
    df = df_original.copy()

    for diff_days in range(0, 1):
        if diff_days == 1:
            return None

        if MODE == 'test':
            df = df_original.iloc[:-diff_days].copy()

        if df is None or df.empty or len(df) < 70:
            continue

        average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
        if average_volume < 300:  # only take shares with 1M+ average volume
            continue

        current_price = df['Close'].tolist()[-1]
        if 1 > current_price or current_price > 100:
            continue  # ignore penny stocks and huge stocks

        lows = df['Low'].tolist()
        highs = df['High'].tolist()
        open_prices = df['Open'].tolist()
        close_prices = df['Close'].tolist()

        atr = calculate_atr(df)
        if check_for_bad_candles(df, atr):
            continue

        border = 10
        levels = []

        # Find local minima and maxima indices
        minima_idx = argrelextrema(df['Low'].values, np.less, order=border)[0]
        maxima_idx = argrelextrema(df['High'].values, np.greater, order=border)[0]

        # Get corresponding data points
        local_minima = df.iloc[minima_idx]
        local_maxima = df.iloc[maxima_idx]

        second_levels = []

        for i, (index, row) in enumerate(local_minima.iterrows()):
            levels.append({'index': index, 'v': row['Low']})
            second_levels.append(row['Low'])

        for i, (index, row) in enumerate(local_maxima.iterrows()):
            levels.append({'index': index, 'v': row['High']})
            second_levels.append(row['High'])

        if len(levels) >= 2:
            # level = sorted(levels, key=lambda x: x['index'])[-2]['v']

            for level_ in sorted(levels, key=lambda x: x['index'])[-4:]:
                level = level_['v']

                f = 0
                for i, (index, row) in enumerate(df.iterrows()):
                    if row['Low'] < level < row['High']:
                        f += 1

                if f < 3 and abs(close_prices[-1] - level) < 0.1 * atr:
                    # start_index = max(-50 - diff_days + 5, 1 - len(df_original))
                    # df_draw = df_original.iloc[start_index:-diff_days+5].copy()
                    df_draw = df_original.iloc[-50:].copy()
                    df_draw.index = df_draw.index.strftime('%b %d')

                    if len(df_draw) > 0:
                        draw(df_draw, file_name=f'{ticker}', ticker=ticker, boxes=[], level=level,
                             future=0, second_levels=second_levels)


print('Starting threads...')
Parallel(n_jobs=10, require='sharedmem', timeout=200)(delayed(run_me)(ticker) for ticker in TICKERS)
