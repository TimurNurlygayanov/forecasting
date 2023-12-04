
from bot.utils import get_data
from bot.utils import get_tickers_polygon
from crypto_forex.utils import ALL_TICKERS

from joblib import Parallel, delayed


TICKERS = ALL_TICKERS  # get_tickers_polygon(limit=5000)  # 2000
RESULTS = set()


def run_me(ticker):
    global RESULTS

    df = get_data(ticker, period='minute', multiplier=15, days=5)

    if df is None or len(df) < 200:
        return None

    # print(df)

    high = df['High'].tolist()
    close_price = df['Close'].tolist()
    low = df['Low'].tolist()

    total_period_len = 15
    for start_point in range(5, 100, 5):
        start, end = start_point, start_point + total_period_len

        medium_high = sum(high[-end:-start]) / total_period_len
        medium_size = sum([high[-i] - low[-i] for i in range(start, end)]) / total_period_len

        if medium_size <= 0:
            print('zero medium size', ticker)
            continue

        bars_near_level = sum([1 for i in range(start, end) if abs(high[-i] - medium_high) / medium_size < 0.001])

        # print(ticker, medium_high, bars_near_level)

        delimiter = (total_period_len // 3)
        first_part = sum([high[-i] - low[-i] for i in range(start, start + delimiter)]) / delimiter
        medium_part = sum([high[-i] - low[-i] for i in range(start + delimiter, start + 2*delimiter)]) / delimiter
        last_part = sum([high[-i] - low[-i] for i in range(start + 2*delimiter, end)]) / delimiter

        # print(ticker, delimiter, first_part, medium_part, last_part)

        if bars_near_level > total_period_len * 0.8:
            RESULTS.add(ticker + str(start) + " , " + str(end))
            if first_part < medium_part < last_part:
                RESULTS.add(ticker + str(start) + " , " + str(end))

    print(RESULTS)


Parallel(n_jobs=-1, require='sharedmem', timeout=20)(delayed(run_me)(ticker) for ticker in TICKERS)

print(RESULTS)
