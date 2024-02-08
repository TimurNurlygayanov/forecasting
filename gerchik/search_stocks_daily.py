
from bot.utils import get_data
from bot.utils import get_tickers_polygon

import numpy as np
from scipy.signal import argrelextrema

from joblib import Parallel, delayed
from utils import draw
from utils import calculate_atr
from utils import find_nakoplenie


TICKERS = get_tickers_polygon(limit=5000)  # 2000
RESULTS = []
BAD_RESULTS = []


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


def run_me(ticker):
    global RESULTS
    global BAD_RESULTS

    diff_days = 20

    df_original = get_data(ticker, period='day', days=200, save_data=False)
    df_original.index = df_original.index.strftime('%b %d')

    df = df_original.iloc[:-diff_days].copy()

    if df is None or df.empty or len(df) < 20:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return None

    current_price = df['Close'].tolist()[-1]
    if 1 > current_price or current_price > 100:
        return None  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return None

    # df.ta.ema(length=9, append=True, col_names=('EMA9',))  # i use ema9 for stop loss on daily timeframe

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
        for i in range(0, 30):
            if lows[-i] < level < highs[-i]:
                k += 1

        if k < 2:
            # Then check if we have some touch of the level
            if abs(current_price - level) < 3 * luft:
                found_signal = True
                selected_level = level
            else:
                if abs(lows[-1] - level) < luft:
                    found_signal = True
                    selected_level = level

                if abs(highs[-1] - level) < luft:
                    found_signal = True
                    selected_level = level

    """
    # make sure we filter bad setups:
    df.ta.ema(length=21, append=True, col_names=('EMA21',))
    ema_pos = df['EMA21'].tolist()[-1] - selected_level

    price_pos = df['Close'].tolist()[-1] - selected_level

    if (ema_pos > 0 > price_pos) or (ema_pos < 0 < price_pos):
        found_signal = False
    ####
    """

    if found_signal:
        df_verification = df_original.iloc[-diff_days:]

        for p in ['Low', 'High']:
            stop_loss_level = df[p].tolist()[-1]  # take low or high of the previous day candle
                                                  # as stop loss
            if p == 'Low':
                stop_loss_level -= luft
            else:
                stop_loss_level += luft

            if stop_loss_level < selected_level:
                take_profit_level = selected_level + 3 * abs(selected_level + luft - stop_loss_level)
            else:
                take_profit_level = selected_level - 3 * abs(selected_level + luft - stop_loss_level)

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

            # Show only good setups:
            if deal_passed:
                print(f'{ticker} level {selected_level}')

                # boxes = find_nakoplenie(df.iloc[-70+diff_days:].copy(), atr=atr)
                # s = check_scenario(df, selected_level)
                # boxes.append(s)
                boxes = []

                RESULTS.append(ticker)

                draw(
                    df_original.iloc[-70:].copy(), file_name=ticker, ticker=ticker,
                    level=selected_level, boxes=boxes
                )
            if deal_failed:
                print(f'{ticker} failed')
                BAD_RESULTS.append(ticker)

                # boxes = find_nakoplenie(df.iloc[-70 + diff_days:].copy(), atr=atr)
                # s = check_scenario(df, selected_level)
                # boxes.append(s)
                boxes = []

                draw(
                    df_original.iloc[-70:].copy(), file_name=ticker, ticker=ticker,
                    level=selected_level, boxes=boxes
                )



print('Starting threads...')
Parallel(n_jobs=10, require='sharedmem', timeout=200)(delayed(run_me)(ticker) for ticker in TICKERS)

print('= ' * 20)

for t in RESULTS:
    print(t)

BAD_RESULTS = list(set(BAD_RESULTS))
BAD_RESULTS = [s for s in BAD_RESULTS if s not in RESULTS]

print(f'{len(RESULTS)} of good trades, {len(BAD_RESULTS)} of bad trades')
print(100 * len(RESULTS) / (len(RESULTS + BAD_RESULTS)), '% of successful trades')
