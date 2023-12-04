
from bot.utils import get_data
from bot.utils import get_tickers_polygon

from tqdm import tqdm
from scipy.signal import argrelextrema
import numpy as np


TICKERS = get_tickers_polygon(limit=5000)  # 2000
RESULTS = {}


def check_for_gaps(df):
    gaps = 0

    for i, (index, row) in enumerate(df.iterrows()):
        if i > 0:
            if row['High'] - row['Low'] < 0.4 * row['ATR']:
                gaps += 1

    if gaps / len(df) > 0.3:
        return True

    return False


def check_recent_activity(closed_prices, level_price, current_price):
    all_good = True
    if current_price < level_price:
        for p in closed_prices:
            if p > level_price:
                all_good = False
    elif current_price > level_price:
        for p in closed_prices:
            if p < level_price:
                all_good = False

    return all_good


# TICKERS = TICKERS[:200]
for ticker in tqdm(TICKERS):
    df = get_data(ticker, period='day', days=100)
    df['ATR'] = 0

    if df is None or df.empty or len(df) < 20:
        continue

    df.ta.atr(append=True, col_names=('ATR',))
    # if check_for_gaps(df):
    #     continue

    current_price = df['Close'].tolist()[-1]
    delta_price = 0.3 * df['ATR'].tolist()[-1]

    if 1 > current_price or current_price > 100:
        continue  # ignore penny stocks and huge stocks

    # Make sure the ticker has some movement
    if 'ATR' not in df:
        continue

    atr_is_decreasing = False
    if df['ATR'].tolist()[-1] < df['ATR'].tolist()[-2] < df['ATR'].tolist()[-5]:
        atr_is_decreasing = True

    if not atr_is_decreasing:
        continue  # make sure the bars size descreases when the price if going near the level

    price_diff = 0.01  # 0.1 * df['ATR'].tolist()[-1]

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000

    if average_volume < 100:  # ignore shares without volume
        continue

    RESULTS[ticker] = {'limit': set(), 'mirror': set(), 'trend_reversal': set()}
    levels_found = 0
    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    # find local mins and maxs
    df['min'] = df.iloc[argrelextrema(df.Low.values[:-2], np.less_equal,
                                      order=5)[0]]['Low']
    df['max'] = df.iloc[argrelextrema(df.High.values[:-2], np.greater_equal,
                                      order=5)[0]]['High']

    mins = [a for a in df['min'].tolist() if a > 0]
    maxs = [a for a in df['max'].tolist() if a > 0]

    for price_level in mins + maxs:
        if abs(price_level - current_price) < delta_price:
            if check_recent_activity(df['Close'].values[-20:], level_price=price_level, current_price=current_price):
                RESULTS[ticker]['trend_reversal'].add(price_level)
                levels_found += 1

    for level_type in ['limit', 'mirror']:
        prices = highs + lows
        bars_required = 2

        if level_type == 'mirror':
            prices = sorted(prices)
            bars_required = 5

        group = []
        limit_levels = set()
        previous_price = prices[0]

        for p in prices:
            if 100 * abs(previous_price - p) / p < price_diff:
                group.append(p)
            else:
                if len(group) >= bars_required:
                    level = sum(group) / len(group)  # x_round(sum(group) / len(group))

                    # if price is near the level with 2% diff, add it to the list
                    if abs(level - current_price) < delta_price:
                        if check_recent_activity(df['Close'].values[-20:], level_price=level,
                                                 current_price=current_price):
                            limit_levels.add(level)
                            levels_found += 1

                group = []

            previous_price = p

        if limit_levels:
            RESULTS[ticker][level_type] = limit_levels

    if not levels_found:
        del RESULTS[ticker]

    for t in RESULTS:
        res = ''
        for level_type in RESULTS[t]:
            if RESULTS[t][level_type]:
                res += f' {level_type} {RESULTS[t][level_type]}'

        print(f' {t}  ', res)

    print(len(RESULTS))
