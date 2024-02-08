from bot.utils import get_data
from bot.utils import get_tickers_polygon

from joblib import Parallel, delayed
from utils import draw
from utils import calculate_atr
from utils import find_nakoplenie


TICKERS = get_tickers_polygon(limit=5000)  # 2000


def run_me(ticker):

    df = get_data(ticker, period='day', days=100, save_data=False)
    df.index = df.index.strftime('%b %d')

    if df is None or df.empty or len(df) < 20:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return None

    current_price = df['Close'].tolist()[-1]
    if 1 > current_price or current_price > 100:
        return None  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)

    boxes = find_nakoplenie(df.iloc[:-10].copy(), atr=atr)

    d = False
    for b in boxes:
        if b['end_int'] == len(df.iloc[:-10]) - 2:
            d = True

    if d:
        draw(df, file_name=ticker, ticker=ticker, boxes=boxes)


print('Starting threads...')
Parallel(n_jobs=10, require='sharedmem', timeout=200)(delayed(run_me)(ticker) for ticker in TICKERS)

