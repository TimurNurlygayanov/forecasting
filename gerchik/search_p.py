import uuid

from bot.utils import get_data
from bot.utils import get_tickers_polygon

from joblib import Parallel, delayed
from utils import draw
from utils import detect


TICKERS = get_tickers_polygon(limit=5000)


def run_me(ticker):
    global RESULTS

    df = get_data(ticker, period='day', days=50, save_data=False)

    if df is None or df.empty or len(df) < 20:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return None

    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    opens = df['Open'].tolist()
    closes = df['Close'].tolist()
    s1 = highs[-1] - lows[-1]
    s2 = highs[-2] - lows[-2]
    s3 = highs[-3] - lows[-3]

    if s1 < s2 < s3:   # volatilnost padaet
        if lows[-1] > lows[-2] > lows[-3]:
            if opens[-1] < closes[-1]:
                k = 0
                for high in highs[-10:-1]:
                    if abs(high - highs[-1]) < 0.03:
                        k += 1

                if k >= 2:
                    draw(df, file_name=ticker, ticker=ticker, level=highs[-1])

        if highs[-1] < highs[-2] < highs[-3]:
            if opens[-1] > closes[-1]:
                k = 0
                for low in lows[-10:-1]:
                    if abs(low - lows[-1]) < 0.03:
                        k += 1

                if k >= 2:
                    draw(df, file_name=ticker, ticker=ticker, level=lows[-1])

    # result = detect(f'training_data/{ticker}.png', train=True)


print('Starting threads...')
Parallel(n_jobs=20, require='sharedmem', timeout=20)(delayed(run_me)(ticker) for ticker in TICKERS)
# run_me('WW')
