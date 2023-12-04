import time

from bot.utils import get_data
from crypto_forex.utils import ALL_TICKERS

from joblib import Parallel, delayed


TICKERS = ALL_TICKERS
RESULTS = []

# Just save all daily lows and highs to make sure we got BSU from daily level
DATA = {}
DATA_RAW = {}
for ticker in TICKERS:
    print(f'Collecting daily data for {ticker}')

    df = get_data(ticker, period='day', multiplier=1, save_data=True, days=200)

    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    DATA[ticker] = sorted(lows + highs)
    DATA_RAW[ticker] = df


def search_for_bsu(ticker, bsu_price, luft):
    for level in DATA[ticker]:
        if abs(level - bsu_price) < 0.1 * luft:
            return True

    return False


def calculate_atr(ticker):
    df = DATA_RAW[ticker]

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


def run_me(ticker):
    global RESULTS

    atr = calculate_atr(ticker)

    for period, multiplier in [('minute', 30), ('hour', 1)]:
        days = 2
        if period == 'hour':
            days = 20

        print(f'Collecting data for {ticker} {multiplier} {period}...')
        df = get_data(ticker, period=period, multiplier=multiplier, save_data=False, days=days)

        if df is None or df.empty or len(df) < 20:
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

                # Check if we confirm this level on daily timeframe:
                if search_for_bsu(ticker=ticker, bsu_price=highs[-2], luft=luft):
                    found_signal = True
                    order_price = lows[-2] + luft
                    stop_price = order_price - stop_loss

        if abs(highs[-1] - highs[-2]) < luft:
            if highs[-2] >= highs[-1]:

                # Check if we confirm this level on daily timeframe:
                if search_for_bsu(ticker=ticker, bsu_price=highs[-2], luft=luft):
                    found_signal = True
                    order_price = highs[-2] - luft
                    stop_price = order_price + stop_loss

        if found_signal:
            RESULTS.append(f'{ticker} limit order: {order_price:.2f}, stop: {stop_price:.2f}'
                           f' on {multiplier} {period} timeframe')
            print(len(RESULTS), RESULTS)


Parallel(n_jobs=2, require='sharedmem', timeout=60)(delayed(run_me)(ticker) for ticker in TICKERS)

for t in RESULTS:
    print(t)

print(len(RESULTS))
