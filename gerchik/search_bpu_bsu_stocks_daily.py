
from bot.utils import get_data
from bot.utils import get_tickers_polygon

from joblib import Parallel, delayed


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


def run_me(ticker):
    global RESULTS

    df = get_data(ticker, period='day', days=200, save_data=False)

    if df is None or df.empty or len(df) < 50:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 1000:  # only take shares with 1M+ average volume
        return None

    current_price = df['Close'].tolist()[-1]
    if 3 > current_price or current_price > 300:
        return None  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return None

    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    # open_prices = df['Open'].tolist()
    # close_prices = df['Close'].tolist()
    stop_loss = 0.1 * atr
    luft = 0.02 * atr
    order_price = 0
    stop_price = 0

    found_signal = False
    if abs(lows[-1] - lows[-2]) < luft:
        if lows[-2] <= lows[-1]:
            # check if we close near to the level:
            # if close_prices[-1] < open_prices[-1]:
            #     if abs(close_prices[-1] - lows[-1]) < 0.3 * abs(close_prices[-1] - highs[-1]):

            if search_for_bsu(lows=lows, highs=highs, bsu_price=lows[-2], luft=luft):
                found_signal = True
                order_price = lows[-2] + luft
                stop_price = order_price - stop_loss

    if abs(highs[-1] - highs[-2]) < luft:
        if highs[-2] >= highs[-1]:
            # if close_prices[-1] > open_prices[-1]:
            #     if abs(close_prices[-1] - highs[-1]) < 0.3 * abs(close_prices[-1] - lows[-1]):

            if search_for_bsu(lows=lows, highs=highs, bsu_price=highs[-2], luft=luft):
                found_signal = True
                order_price = highs[-2] - luft
                stop_price = order_price + stop_loss

    if found_signal:
        print(f'{ticker} limit order: {order_price:.2f}, stop: {stop_price:.2f}')
        RESULTS.append(f'{ticker} limit order: {order_price:.2f}, stop: {stop_price:.2f}')
        # print(len(RESULTS), RESULTS)


Parallel(n_jobs=-1, require='sharedmem', timeout=20)(delayed(run_me)(ticker) for ticker in TICKERS)

for t in RESULTS:
    print(t)

print(len(RESULTS))
