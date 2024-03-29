
import warnings

warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings("ignore", message="urllib3")

from bot.utils import get_data
from bot.utils import get_tickers_polygon

import numpy as np
from scipy.signal import argrelextrema

from joblib import Parallel, delayed
from utils import draw
from utils import calculate_atr
from utils import find_nakoplenie
from utils import check_for_bad_candles
from utils import search_for_bsu
from utils import check_podzhatie
from utils import check_simple_lp


def run_me(ticker, progress=0):
    print(f'  Loading data... {progress:.2f}% done     ', end='\r')

    try:
        # take additional 150 days to identify levels properly
        df = get_data(ticker, period='day', days=150, save_data=False)
        df.index = df.index.strftime('%b %d')
    except Exception as e:
        print(e)
        return ticker, 0

    if df is None or df.empty or len(df) < 50:
        return ticker, 0

    df_original = df.copy()
    df = df.iloc[:-5]  # cut the last day to check

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return ticker, 0

    current_price = df['Close'].tolist()[-1]
    if 1 > current_price or current_price > 100:
        return ticker, 0  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return ticker, 0

    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    open_prices = df['Open'].tolist()
    close_prices = df['Close'].tolist()

    stop_loss = 0.1 * atr
    luft = 0.02 * atr
    order_price = 0
    stop_price = 0

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
                levels.append(level_h)
            if level_l:
                levels.append(level_l)

    if abs(highs[-1] - highs[-2]) < 0.02 or abs(highs[-1] - highs[-3]) < 0.02 or abs(highs[-1] - highs[-4]) < 0.02:
        levels.append(highs[-1])
    if abs(lows[-1] - lows[-2]) < 0.02 or abs(lows[-1] - lows[-3]) < 0.02 or abs(lows[-1] - lows[-4]) < 0.02:
        levels.append(lows[-1])

    #  limit + mirror levels search

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

                levels.append(level)

            group = []

        previous_price = p

    # izlom trenda search

    border = 10

    # Find local minima and maxima indices
    minima_idx = argrelextrema(df['Low'].values, np.less, order=border)[0]
    maxima_idx = argrelextrema(df['High'].values, np.greater, order=border)[0]

    # Get corresponding data points
    local_minima = df.iloc[minima_idx]
    local_maxima = df.iloc[maxima_idx]

    for i, (index, row) in enumerate(local_minima.iterrows()):
        levels.append(row['Low'])

    for i, (index, row) in enumerate(local_maxima.iterrows()):
        levels.append(row['High'])

    level = check_podzhatie(df)
    if level > 0:
        levels.append(level)

    # Choosing the right level:

    selected_level = 0
    for level in levels:
        # Check if level is clear:

        k = 0
        for i in range(0, len(df)):
            if lows[-i] < level < highs[-i]:
                k += 1

            if lows[-i] > level:
                k += 1

        if k < 2:
            if highs[-1] > level and open_prices[-1] < level and close_prices[-1] < level:
                found_signal = True
                selected_level = level

    if found_signal:
        buy_price = selected_level
        stop_loss = selected_level + 0.2 * atr
        take_profit = selected_level - 7 * abs(buy_price - stop_loss)

        # If we didn't spend enough fuel (ATR) or we spent too much - do not trade this
        previous_close = df['Close'].values[-2]
        proshli_do_urovnia = 100 * abs(selected_level - previous_close) / atr
        if proshli_do_urovnia < 50 or proshli_do_urovnia > 300:
            return ticker, 0
        ####

        boxes = [
            check_simple_lp(
                df, selected_level, atr,
                buy_price, stop_loss, take_profit
            )
        ]  # [check_scenario(df, selected_level)]
        levels_to_draw = []

        draw(
            df_original.iloc[-70:].copy(), file_name=f'{ticker}', ticker=ticker,
            level=selected_level, boxes=boxes, second_levels=levels_to_draw, future=0,
            buy_price=buy_price, stop_loss=stop_loss, take_profit=take_profit, buy_index=df.index.values[-1],
            zig_zag=False,
        )

        """
        try:
            df_hour = get_data(ticker, period='hour', days=10, save_data=False)

            draw(
                df_hour, file_name=f'{ticker}_h', ticker=ticker,
                level=selected_level, boxes=[], second_levels=levels_to_draw, future=0,
                zig_zag=True,
            )
        except:
            pass
        """

    return ticker, 0


if __name__ == "__main__":
    print('Starting threads...')
    TICKERS = get_tickers_polygon(limit=5000)  # 2000
    total_results = []

    Parallel(n_jobs=-1, max_nbytes='200M', backend="multiprocessing", timeout=100)(
        delayed(run_me)(ticker, 100*i/len(TICKERS)) for i, ticker in enumerate(TICKERS)
    )

    print()
