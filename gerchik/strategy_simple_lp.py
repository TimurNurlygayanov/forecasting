
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


def check_for_bad_candles(df, atr):
    gaps = 0

    for i, (index, row) in enumerate(df.iterrows()):
        if i > 0:
            if abs(row['High'] - row['Low']) < 0.4 * atr:
                gaps += 1
            if abs(row['High'] - row['Low']) < 0.1:
                gaps += 1

    if gaps / len(df) > 0.05:
        return True

    return False


def search_for_bsu(lows, highs, bsu_price, luft):
    for i in range(len(lows) - 3):
        if abs(lows[i] - bsu_price) < 0.1 * luft:
            return True
        if abs(highs[i] - bsu_price) < 0.1 * luft:
            return True

    return False


def check_dozhatie(df):
    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    opens = df['Open'].tolist()
    closes = df['Close'].tolist()
    s1 = highs[-1] - lows[-1]
    s2 = highs[-2] - lows[-2]
    s3 = highs[-3] - lows[-3]

    delta = 0.03

    if s1 < s2 < s3:   # volatilnost padaet
        if lows[-1] > lows[-2] > lows[-3]:
            if opens[-1] < closes[-1]:
                # Check for the confirmation
                k = 0
                for high in highs[-10:-1]:
                    if abs(high - highs[-1]) <= delta:
                        k += 1

                if k >= 2:
                    return highs[-1]  # draw(df, file_name=ticker, ticker=ticker, level=highs[-1])

        if highs[-1] < highs[-2] < highs[-3]:
            if opens[-1] > closes[-1]:
                k = 0
                for low in lows[-10:-1]:
                    if abs(low - lows[-1]) <= delta:
                        k += 1

                if k >= 2:
                    return lows[-1]  # draw(df, file_name=ticker, ticker=ticker, level=lows[-1])

    return 0


def check_scenario(df, level):
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    current_close = df['Close'].tolist()[-1]

    last_candle_size = highs[-1] - lows[-1]
    ratio = abs(level - current_close) / last_candle_size

    blizhnii_retest = False
    for i in range(4, 12):
        if lows[-i] < level < highs[-i]:
            blizhnii_retest = True

    label = f'Ratio: {round(ratio, 2):.2f}'
    if blizhnii_retest:
        label += '<br> Ближний ретест'
    else:
        label += '<br> Дальний ретест'

    return {
        'x0': df.index[-3], 'x1': df.index[-1], 'y0': min(lows[-3:]), 'y1': max(highs[-3:]),
        'label': label, 'color': 'rgba(55,200,34,0.2)'
    }


def run_me(ticker, diff_days=30, progress=0):
    print(f'  Loading data... {progress:.2f}% done     ', end='\r')

    try:
        # take additional 150 days to identify levels properly
        df_original = get_data(ticker, period='day', days=diff_days + 150, save_data=False)
        df_original.index = df_original.index.strftime('%b %d')
    except Exception as e:
        print(e)
        return ticker, 0

    if df_original is None or df_original.empty or len(df_original) < diff_days + 50:
        return ticker, 0

    df = df_original.iloc[:-diff_days].copy()

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

    level = check_dozhatie(df)
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
        df_verification = df_original.iloc[-diff_days:]

        low_low = df_original[-70:]['Low'].min()
        high_high = df_original[-70:]['High'].max()
        levels_to_draw = [level for level in levels if low_low <= level <= high_high]

        buy_price = df_original['Open'].values[-diff_days]
        stop_loss_level = selected_level + atr / 2  # df['High'].tolist()[-1] + abs(selected_level - df['High'].tolist()[-1])
        take_profit_level = buy_price - 4 * abs(buy_price - stop_loss_level)

        # experimental:
        buy_price = selected_level
        stop_loss_level = selected_level + atr * 0.1  # df['High'].tolist()[-1] + abs(selected_level - df['High'].tolist()[-1])
        take_profit_level = buy_price - 3 * abs(buy_price - stop_loss_level)
        #

        # Only take small stops
        if 100 * abs(buy_price - stop_loss_level) / buy_price > 3:
            return ticker, 0

        # Make sure we only take correct trades
        if buy_price > selected_level:
            return ticker, 0
        if open_prices[-2] > selected_level or close_prices[-2] > selected_level:
            return ticker, 0

        deal_failed = False
        deal_passed = False
        for i, (index, row) in enumerate(df_verification.iterrows()):
            if not deal_passed and not deal_failed:
                if row['Low'] < stop_loss_level < row['High']:
                    deal_failed = True
                if row['Low'] > stop_loss_level > row['High']:
                    deal_failed = True

                if not deal_failed:
                    if row['Low'] < take_profit_level < row['High']:
                        deal_passed = True
                    if row['Low'] > take_profit_level > row['High']:
                        deal_passed = True

        if deal_passed:
            # boxes = find_nakoplenie(df.iloc[-70+diff_days:].copy(), atr=atr)
            # s = check_scenario(df, selected_level)
            # boxes.append(s)
            boxes = []

            draw(
                df_original.iloc[-70:].copy(), file_name=f'{ticker}_{diff_days}_passed', ticker=ticker,
                level=selected_level, boxes=boxes, second_levels=levels_to_draw, future=0,
                zig_zag=True, buy_index=df_original.index.values[-diff_days],
                buy_price=buy_price, stop_loss=stop_loss_level, take_profit=take_profit_level
            )

            return ticker, 1
        else:
            # boxes = find_nakoplenie(df.iloc[-70 + diff_days:].copy(), atr=atr)
            # s = check_scenario(df, selected_level)
            # boxes.append(s)
            boxes = []

            draw(
                df_original.iloc[-70:].copy(), file_name=f'{ticker}_{diff_days}_failed', ticker=ticker,
                level=selected_level, boxes=boxes, second_levels=levels_to_draw, future=diff_days,
                zig_zag=True, buy_index=df_original.index.values[-diff_days],
                buy_price=buy_price, stop_loss=stop_loss_level, take_profit=take_profit_level
            )

            return ticker, -1

    return ticker, 0


if __name__ == "__main__":
    print('Starting threads...')
    TICKERS = get_tickers_polygon(limit=5000)  # 2000
    total_results = []

    for days in range(30, 35):
        # max_nbytes='200M',
        RESULTS = Parallel(n_jobs=7, backend="multiprocessing", timeout=100)(
            delayed(run_me)(ticker, days, 100*i/len(TICKERS)) for i, ticker in enumerate(TICKERS)
        )

        print('= ' * 20)

        good_results = [r for r in RESULTS if r[1] > 0]
        bad_results = [r for r in RESULTS if r[1] < 0]

        print(f'Day {days}')
        print(f'{len(good_results)} of good trades, {len(bad_results)} of bad trades')

        if len(good_results + bad_results) > 0:
            print(f'{100 * len(good_results) / (len(good_results + bad_results)):.2f}% of successful trades')

        print('= ' * 20)
        print()

        total_results += good_results + bad_results

    print('* ' * 20)
    good_results = [r for r in total_results if r[1] > 0]
    bad_results = [r for r in total_results if r[1] < 0]
    print('Summary statistic:')
    print(f'{len(good_results)} of good trades, {len(bad_results)} of bad trades')
    print(f'{100 * len(good_results) / (len(good_results + bad_results)):.2f}% of successful trades')
