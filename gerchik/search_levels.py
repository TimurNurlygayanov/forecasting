
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


# TICKERS = TICKERS[:200]
for ticker in tqdm(TICKERS):
    df = get_data(ticker, period='day', days=100)
    df['ATR'] = 0

    if df is None or df.empty or len(df) < 20:
        continue

    df.ta.atr(append=True, col_names=('ATR',))
    if check_for_gaps(df):
        continue

    current_price = df['Close'].tolist()[-1]

    if 3 > current_price or current_price > 100:
        continue  # ignore penny stocks and huge stocks

    # Make sure the ticker has some movement
    # if 'ATR' not in df or df['ATR'].tolist()[-1] < 0.5:
    #     continue

    price_diff = 0.02  # 0.1 * df['ATR'].tolist()[-1]

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000

    if average_volume < 200:
        continue

    RESULTS[ticker] = {'limit': set(), 'mirror': set(), 'trend_reversal': set()}
    levels_found = 0
    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    df['min'] = df.iloc[argrelextrema(df.Low.values, np.less_equal,
                                      order=10)[0]]['Open']
    df['max'] = df.iloc[argrelextrema(df.High.values, np.greater_equal,
                                      order=10)[0]]['Close']

    mins = [a for a in df['min'].tolist() if a > 0]
    maxs = [a for a in df['max'].tolist() if a > 0]

    for price_level in mins + maxs:
        if abs(price_level - current_price) < price_diff:
            RESULTS[ticker]['trend_reversal'].add(price_level)
            levels_found += 1


    for level_type in ['limit']:  # 'mirror'
        prices = highs + lows

        if level_type == 'mirror':
            prices = sorted(prices)

        group = []
        limit_levels = set()
        previous_price = prices[0]

        for p in prices:
            if 100 * abs(previous_price - p) / p < price_diff:
                group.append(p)
            else:
                if len(group) > 5:
                    level = sum(group) / len(group)  # x_round(sum(group) / len(group))

                    # if price is near the level with 2% diff, add it to the list
                    if abs(level - current_price) < price_diff:
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
