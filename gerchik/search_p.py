import uuid

from bot.utils import get_data
from bot.utils import get_tickers_polygon

from joblib import Parallel, delayed
from utils import draw
from utils import detect


TICKERS = get_tickers_polygon(limit=5000)


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


def run_me(ticker):
    global RESULTS

    df = get_data(ticker, period='day', days=70, save_data=False)
    # df = df.iloc[:-10].copy()

    if df is None or df.empty or len(df) < 20:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 300k+ average volume
        return None

    current_price = df['Close'].tolist()[-1]
    if 1 > current_price or current_price > 100:
        return None  # ignore penny stocks and huge stocks

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
                    draw(df, file_name=ticker, ticker=ticker, level=highs[-1])

        if highs[-1] < highs[-2] < highs[-3]:
            if opens[-1] > closes[-1]:
                k = 0
                for low in lows[-10:-1]:
                    if abs(low - lows[-1]) <= delta:
                        k += 1

                if k >= 1:
                    draw(df, file_name=ticker, ticker=ticker, level=lows[-1])


print('Starting threads...')
Parallel(n_jobs=20, require='sharedmem', timeout=20)(delayed(run_me)(ticker) for ticker in TICKERS)
