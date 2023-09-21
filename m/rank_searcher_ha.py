
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic

from tqdm import tqdm

from bot.utils import get_data
from bot.utils import get_tickers_polygon


PERIOD = 30
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
    ha_row2 = dfHA.iloc[-2]
    ha_row3 = dfHA.iloc[-3]
    ha_row4 = dfHA.iloc[-4]

    red_area_before = ha_row4['Close'] < ha_row4['Open']

    ha_row_previous3_body = abs(ha_row3['Close'] - ha_row3['Open'])
    ha_row_previous3_shade = abs(ha_row3['High'] - ha_row3['Low'])
    dodge_green = ha_row3['Close'] > ha_row3['Open']
    dodge_signal = ha_row_previous3_shade / ha_row_previous3_body > 10 and dodge_green

    previous_strong = ha_row2['Open'] == ha_row2['Low'] and ha_row2['Open'] < ha_row2['Close']
    current_strong = ha_row['Open'] == ha_row['Low'] and ha_row['Open'] < ha_row['Close']

    ha_rows7 = dfHA.tail(7)
    res = [0] * 7
    for i, (index, ha_row_x) in enumerate(ha_rows7.iterrows()):
        if ha_row_x['Close'] > ha_row_x['Open']:
            res[i] = 1

    if row['ATR'] / row['Close'] < 0.01:
        return None

    stop_loss = row['Low'] - 1.1 * row['ATR']

    if dodge_signal and previous_strong and current_strong and red_area_before:
        # if res == [1, 1, 1, 1, 1, 0, 1] and row['High'] > df.iloc[-2]['High']:
        # if row['S_trend_d'] > 0:
        if stop_loss / row['Close'] > 0.9:
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
