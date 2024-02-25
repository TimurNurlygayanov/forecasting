# based on this idea https://www.youtube.com/watch?v=3zI_l_P-lF8
#

import warnings
warnings.filterwarnings("ignore")

# from crypto_forex.utils import ALL_TICKERS
# from app.utils import get_data_alpha

from datetime import datetime

import json
from joblib import Parallel, delayed

import os
import pandas as pd
from numpy_ext import rolling_apply

from bot.utils import get_tickers_polygon
from bot.utils import get_data

from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles
from gerchik.utils import draw


results = {}


def run_me(ticker):
    global results

    results[ticker] = []

    file_name = f'collect_data/calculated/h_{ticker}.parquet'

    if os.path.isfile(file_name):
        df = pd.read_parquet(file_name)
        df = df.iloc[:-100].copy()

        df.ta.ema(close='Low', length=3, append=True, col_names=(f'EMA3',))  # we take this instead of price itself
        df['EMA3_prev'] = df['EMA3'].shift(1)

        for e in range(50, 200, 10):
            df.ta.ema(length=e, append=True, col_names=(f'EMA{e}',))
            df[f'EMA{e}_prev'] = df[f'EMA{e}'].shift(1)

            crossed = 0
            bounce = 0
            break_out = 0

            for i, (index, row) in enumerate(df.iterrows()):
                if i > e:
                    if row['EMA3'] < row[f'EMA{e}'] + row['ATR']/2 and row['EMA3_prev'] > row[f'EMA{e}_prev'] + row['ATR']/2:
                        crossed = 1

                    if crossed:
                        if row['EMA3'] > row[f'EMA{e}'] + row['ATR']/2 and row['EMA3_prev'] < row[f'EMA{e}_prev'] + row['ATR']/2:
                            crossed = 0
                            bounce += 1

                        if row['EMA3'] < row[f'EMA{e}'] - row['ATR']/2 and row['EMA3_prev'] > row[f'EMA{e}_prev'] - row['ATR']/2:
                            crossed = 0
                            break_out += 1

            results[ticker].append({'indicator': f'EMA{e}', 'bounce': bounce, 'break_out': break_out})

            if bounce + break_out > 4:
                if bounce / (bounce + break_out) > 0.5 and bounce > 4:
                    print(f'{ticker} EMA{e} {100 * bounce / (bounce + break_out):.2f} % total {bounce} within {len(df)-e} hours')


if __name__ == "__main__":
    print('Preparing training dataset...')

    TICKERS = get_tickers_polygon(limit=5000)  # this is for shares
    TICKERS = TICKERS[:1000]

    for t in TICKERS:
        run_me(ticker=t)

    with open('collect_data/ema_support_data.txt', encoding='utf-8', mode='w+') as f:
        json.dump(results, f)
