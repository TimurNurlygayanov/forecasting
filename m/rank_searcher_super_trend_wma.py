
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic

from tqdm import tqdm

from bot.utils import get_data
from bot.utils import get_tickers_polygon


PERIOD = 40
SELECTED = []
PASSED = []



def check(ticker='AAPL'):
    df = get_data(ticker)

    check_data = df.tail(PERIOD)
    df = df.iloc[:-PERIOD]

    last_index = df.shape[0] - 1
    if last_index < 200:
        return None

    df.ta.ema(length=200, append=True, col_names=('EMA200',))
    df.ta.wma(length=50, append=True, col_names=('WMA50',))
    df.ta.atr(append=True, period=20, col_names=('ATR',))

    df.ta.supertrend(append=True, length=34, multiplier=3.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    dfHA = df.ta.ha()
    dfHA.rename(columns={'HA_open': 'Open', 'HA_close': 'Close', 'HA_low': 'Low', 'HA_high': 'High'}, inplace=True)
    dfHA.ta.supertrend(append=True, length=34, multiplier=3.0,
                       col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    row = df.iloc[-1]
    ha_row = dfHA.iloc[-1]
    ha_row_previous = dfHA.iloc[-2]

    if row['ATR'] / row['Close'] < 0.01:
        return None

    signal1 = ha_row['Open'] / row['WMA50'] < 1.1
    signal2 = row['WMA50'] / row['EMA200'] < 1.1

    stop_loss = row['Low'] - 1.1 * row['ATR']

    if row['Close'] > row['Open'] > row['WMA50'] > row['EMA200']:
        if row['S_trend_d'] > 0 and ha_row['S_trend_d'] > 0:
            if ha_row['Open'] < ha_row['Close']:    # Green Heikin-Ashi candle
                if ha_row['Open'] == ha_row['Low']:   # Candle without tail in the bottom
                    # if ha_row_previous['Open'] > ha_row_previous['Low']:  # ?
                    if ha_row['Close'] > ha_row_previous['Close']:
                        if signal1 and signal2 and stop_loss / row['Close'] > 0.9:
                            SELECTED.append(ticker)

                            state = 'not ready'
                            for i, (index, check_row) in enumerate(check_data.iterrows()):
                                if state == 'not ready':
                                    if check_row['Low'] < stop_loss:
                                        state = f"failed {100 * (stop_loss / row['Close'] - 1):.2f}"
                                        print(row.name)
                                        break
                                    elif check_row['High'] > row['Close'] + 2 * row['ATR']:
                                        state = f"passed +{100 * 2 * row['ATR'] / row['Close']:.2f}"
                                        PASSED.append(ticker)
                                        break

                            print(ticker, state)
                            # print(row)
                            print('- ' * 20)


if __name__ == '__main__':

    TICKERS = get_tickers_polygon(5000)

    for ticker in tqdm(TICKERS[:2000]):
        check(ticker)

    print(100 * len(PASSED) / len(SELECTED), '% success rate')
