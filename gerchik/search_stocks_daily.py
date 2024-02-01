
from bot.utils import get_data
from bot.utils import get_tickers_polygon

from joblib import Parallel, delayed
from utils import draw



TICKERS = get_tickers_polygon(limit=5000)  # 2000
RESULTS = []


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


def calculate_atr(df):
    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    candles_sizes = []
    for i in range(len(lows)):
        s = abs(highs[i] - lows[i])

        if s > 0:
            candles_sizes.append(s)

    # remove 10 % of smallest candles and 10 % of largest candles
    # so we are getting rid of enormous candles
    candles_sizes_medium = sorted(candles_sizes)[len(lows)//10:-len(lows)//10]
    medium_atr = sum(candles_sizes_medium) / len(candles_sizes_medium)

    # after we calculated medium atr it is time to sort candles one more time
    selected_candles = [s for s in candles_sizes if (medium_atr * 0.5) < s < (1.7 * medium_atr)]
    # now we are ready to provide true ATR:
    return sum(selected_candles[-5:]) / 5


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

    delta = 0.02

    if s1 < s2 < s3:   # volatilnost padaet
        if lows[-1] > lows[-2] > lows[-3]:
            if opens[-1] < closes[-1]:
                # Check for the confirmation
                k = 0
                for high in highs[-10:-1]:
                    if abs(high - highs[-1]) <= delta:
                        k += 1

                if k >= 1:
                    return highs[-1]  # draw(df, file_name=ticker, ticker=ticker, level=highs[-1])

        if highs[-1] < highs[-2] < highs[-3]:
            if opens[-1] > closes[-1]:
                k = 0
                for low in lows[-10:-1]:
                    if abs(low - lows[-1]) <= delta:
                        k += 1

                if k >= 1:
                    return lows[-1]  # draw(df, file_name=ticker, ticker=ticker, level=lows[-1])

    return 0


def run_me(ticker):
    global RESULTS

    df = get_data(ticker, period='day', days=170, save_data=False)

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
    """

    for i, (index, row) in enumerate(df.iterrows()):
        bar_size = row['High'] - row['Low']

        if i < len(df) - 5 and bar_size > 2.3 * atr:
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
    """

    if abs(highs[-1] - highs[-2]) < 0.02 or abs(highs[-1] - highs[-3]) < 0.02 or abs(highs[-1] - highs[-4]) < 0.02:
        levels.append(highs[-1])
    if abs(lows[-1] - lows[-2]) < 0.02 or abs(lows[-1] - lows[-3]) < 0.02 or abs(lows[-1] - lows[-4]) < 0.02:
        levels.append(lows[-1])

    #  limit + mirror levels search

    prices = sorted(highs + lows)
    bars_required = 5

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

    for k in range(20, len(df)-22, 10):
        min_low = min(lows[k-20:k+20])
        max_high = max(highs[k-20:k+20])
        mean_low = sum(lows[k-20:k+20]) / len(lows[k-20:k+20])
        mean_high = sum(highs[k - 20:k + 20]) / len(highs[k - 20:k + 20])

        if abs(min_low - mean_low) > 2 * atr:
            e = 0

            for j in range(k+20, len(df)-1):
                if open_prices[j] > min_low > close_prices[j]:
                    e += 1
                if open_prices[j] < min_low < close_prices[j]:
                    e += 1

            if e < 3:
                levels.append(min_low)

        if abs(max_high - mean_high) > 2 * atr:
            e = 0

            for j in range(k + 20, len(df) - 1):
                if open_prices[j] > max_high > close_prices[j]:
                    e += 1
                if open_prices[j] < max_high < close_prices[j]:
                    e += 1

            if e < 3:
                levels.append(max_high)

    level = check_dozhatie(df)
    if level > 0:
        levels.append(level)

    # Choosing the right level:

    selected_level = 0
    for level in levels:
        if abs(current_price - level) < 2 * luft:
            found_signal = True
            selected_level = level

    if found_signal:
        print(f'{ticker} level {selected_level}')

        RESULTS.append(f'{ticker} level {selected_level}')

        draw(df, file_name=ticker, ticker=ticker, level=selected_level)


print('Starting threads...')
Parallel(n_jobs=20, require='sharedmem', timeout=20)(delayed(run_me)(ticker) for ticker in TICKERS)

print('= ' * 20)

for t in RESULTS:
    print(t)

print(len(RESULTS))
