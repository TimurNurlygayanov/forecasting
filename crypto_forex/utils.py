
import os
import random

import requests

from polygon import RESTClient
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pandas_ta  # for TA magic
from tqdm import tqdm


now = datetime.now()
START = now - timedelta(days=500)
END = now + timedelta(days=1)

# 'X:BTCUSD', 'X:ETHUSD', 'X:LTCUSD',
ALL_TICKERS = [
    'C:AUDCAD', 'C:AUDJPY', 'C:AUDNZD', 'C:AUDUSD', 'C:CADCHF',
    'C:CADJPY', 'C:CHFJPY', 'C:EURAUD', 'C:EURCAD', 'C:EURCHF', 'C:EURGBP',
    'C:EURJPY', 'C:EURNOK', 'C:EURNZD', 'C:EURPLN', 'C:EURSGD',
    'C:EURUSD', 'C:GBPAUD', 'C:GBPCAD', 'C:GBPCHF', 'C:GBPJPY',
    'C:GBPNZD', 'C:GBPUSD', 'C:NZDCAD', 'C:NZDCHF', 'C:NZDJPY', 'C:NZDUSD',
    'C:USDCAD', 'C:USDCHF', 'C:USDJPY', 'C:USDMXN', 'C:USDNOK', 'C:USDSEK',
    'C:USDSGD', 'C:USDZAR'
]
ALL_TICKERS = list(set(ALL_TICKERS))


with open('/Users/timur.nurlygaianov/api_key2.txt', encoding='utf-8', mode='r') as f:
    api_key = f.readlines()
    api_key = ''.join(api_key).strip()


client = RESTClient(api_key=api_key)


def x_round(x):
    return round(x*4)/4


def get_data(ticker='X:BTCUSD', period='hour', multiplier=1, save_data=True, days=50):  # minute
    df = None
    indexes = []
    data = {'Close': [], 'Open': [], 'Low': [], 'High': [], 'vwap': [], 'volume': []}
    START = now - timedelta(days=days)

    try:
        file_name = f'rl/data/{ticker}_hourly_{datetime.now().strftime("%Y-%m-%d")}.xlsx'
        if os.path.isfile(file_name) and save_data:
            df = pd.read_excel(file_name, index_col=0)
        else:
            for a in client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=period,  # "hour"
                                      from_=START.strftime("%Y-%m-%d"),
                                      to=END.strftime("%Y-%m-%d"),
                                      limit=50000):
                date = datetime.fromtimestamp(a.timestamp // 1000).strftime("%Y-%m-%d, %H:%M:%S")

                indexes.append(date)
                data['Close'].append(a.close)
                data['Open'].append(a.open)
                data['High'].append(a.high)
                data['Low'].append(a.low)
                data['vwap'].append(a.vwap)
                data['volume'].append(a.volume)

            df = pd.DataFrame(data, index=indexes)

            df.ta.cdl_pattern(append=True, name=["doji", "morningstar", "hammer", "engulfing", "shootingstar"])

            df.ta.ema(length=7, append=True, col_names=('EMA7',))
            df.ta.ema(length=21, append=True, col_names=('EMA21',))
            df.ta.ema(length=50, append=True, col_names=('EMA50',))
            df.ta.ema(length=200, append=True, col_names=('EMA200',))
            df.ta.supertrend(append=True, length=10, multiplier=3.0,
                             col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
            df.ta.supertrend(append=True, length=12, multiplier=1.5,
                             col_names=('S_trend34', 'S_trend_d34', 'S_trend_l34', 'S_trend_s34',))
            df.ta.rsi(length=14, append=True, col_names=('RSI',))
            df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))
            df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)

            df.ta.atr(append=True, col_names=('ATR',))

            if save_data:
                df.to_excel(file_name, index=True, header=True)
    except Exception as e:
        print(f'No data for {ticker} {e}')

    return df


def is_engulfing_bullish(open1, open2, close1, close2):
    if open1 > close1:   # previous candle is red
        if open2 < close2:  # current candle is green
            if open2 < close1 and close2 > open1:  # green is bigger
                return 1

    return 0


def get_state(df, dfHA, i: int = 0, step_size: int = 10):
    state = [0] * 76
    row = df.iloc[i]

    #
    state[0] = 1 if row['Low'] > row['EMA200'] and df['Low'].values[i - 1] < df['EMA200'].values[i - 1] else 0
    state[1] = 1 if row['Low'] > row['EMA50'] and df['Low'].values[i - 1] < df['EMA50'].values[i - 1] else 0
    state[2] = 1 if row['EMA50'] > row['EMA200'] and df['EMA50'].values[i - 1] < df['EMA200'].values[i - 1] else 0
    state[3] = 1 if row['EMA7'] > row['EMA50'] and df['EMA7'].values[i - 1] < df['EMA50'].values[i - 1] else 0
    state[4] = 1 if row['RSI'] > 30 > df['RSI'].values[i - 1] else 0
    state[5] = 1 if row['RSI'] > 35 > df['RSI'].values[i - 1] else 0
    state[6] = 1 if row['RSI'] > 50 > df['RSI'].values[i - 1] else 0
    state[7] = 1 if row['S_trend_d'] > 0 and row['RSI'] < 50 else 0

    # x custom signals:
    macd_signal = False
    if 0 > row['MACD'] > row['MACD_signal']:
        state[8] = 1

        for j in range(1, 6):
            if df['MACD'].values[i - j] < df['MACD_signal'].values[i - j]:
                macd_signal = True

    if macd_signal and row['S_trend_d'] > 0:
        state[9] = 1 if row['S_trend_d'] > 0 and row['RSI'] < 50 else 0

    #

    is_green = True

    if df['Open'].values[i] < df['Close'].values[i]:
        is_green = False

    if df['Open'].values[i - 1] < df['Close'].values[i - 1]:
        is_green = False

    if df['Open'].values[i - 2] < df['Close'].values[i - 2]:
        is_green = False

    state[10] = 1 if is_green else 0  # last 3 candles are green?

    state[11] = is_engulfing_bullish(
        df['Open'].values[i - 1], df['Open'].values[i],
        df['Close'].values[i - 1], df['Close'].values[i]
    )

    state[12] = 1 if row['RSI'] < 30 else 0
    state[13] = 1 if row['RSI'] < 40 else 0
    state[14] = 1 if row['RSI'] > 70 else 0
    state[15] = 1 if row['S_trend_d'] > 0 else 0
    state[16] = 1 if row['S_trend_d'] > 0 > df['S_trend_d'].values[i - 1] else 0

    state[17] = 1 if row['Low'] > row['EMA200'] else 0
    state[18] = 1 if row['Low'] > row['EMA50'] else 0
    state[19] = 1 if row['EMA50'] > row['EMA200'] else 0

    state[20] = 1 if row['Close'] > row['EMA200'] else 0
    state[21] = 1 if row['Close'] > row['EMA50'] else 0

    # lower low and higher high
    state[22] = 1 if row['Low'] < min(df['Low'].values[i - 10:i - 1]) else 0
    state[23] = 1 if row['Low'] < min(df['Low'].values[i - 50:i - 1]) else 0
    state[24] = 1 if row['Low'] < min(df['Low'].values[i - 200:i - 1]) else 0

    state[25] = 1 if row['High'] > max(df['High'].values[i - 10:i - 1]) else 0
    state[26] = 1 if row['High'] > max(df['High'].values[i - 50:i - 1]) else 0
    state[27] = 1 if row['High'] > max(df['High'].values[i - 200:i - 1]) else 0

    # if price higher that EMA for long time? - EMA 50
    higher_price = True
    for j in range(10):
        if df['Close'].values[i - j] < df['EMA50'].values[i - j]:
            higher_price = False
    state[28] = 1 if higher_price else 0

    # EMA 200
    higher_price = True
    for j in range(10):
        if df['Close'].values[i - j] < df['EMA200'].values[i - j]:
            higher_price = False
    state[29] = 1 if higher_price else 0

    state[30] = 1 if df['EMA200'].values[i] > df['EMA200'].values[i - 10] else 0
    state[31] = 1 if df['EMA50'].values[i] > df['EMA50'].values[i - 10] else 0

    state[32] = 1 if row['MACD_hist'] > df['MACD_hist'].values[i - 1] > df['MACD_hist'].values[i - 2] else 0
    state[33] = 1 if row['MACD_hist'] > 0 else 0

    candle_full = abs(row['High'] - row['Low'])
    candle_body = abs(row['High'] - row['Low'])
    green_hammer = row['Close'] > row['Open'] and (row['Open'] - row['Low']) / candle_full > 0.7
    state[34] = 1 if candle_full > candle_body * 3 and green_hammer else 0

    state[35] = 1 if df['volume'].values[i] > df['volume'].values[i - 1] else 0
    state[36] = 1 if df['volume'].values[i - 1] > df['volume'].values[i - 2] else 0
    state[37] = 1 if df['volume'].values[i - 2] > df['volume'].values[i - 3] else 0

    """
    state[38] = 1 if df['vwap'].values[i] > df['vwap'].values[i - 1] else 0
    state[39] = 1 if df['vwap'].values[i - 1] > df['vwap'].values[i - 2] else 0
    state[40] = 1 if df['vwap'].values[i - 2] > df['vwap'].values[i - 3] else 0
    """

    state[41] = 1 if row['High'] > row['U'] else 0
    state[42] = 1 if row['Close'] > row['U'] else 0

    state[43] = 1 if row['Low'] < row['L'] else 0
    state[44] = 1 if row['Close'] < row['L'] else 0

    state[45] = 1 if df['EMA7'].values[i] > df['EMA7'].values[i - 1] else 0
    state[46] = 1 if df['Close'].values[i] > df['EMA7'].values[i] else 0

    state[47] = 1 if row['S_trend_d34'] > 0 else 0
    state[48] = 1 if row['S_trend_d34'] > 0 > df['S_trend_d34'].values[i - 1] else 0

    state[49] = 1 if row['CDL_DOJI_10_0.1'] > 0 else 0
    state[50] = 1 if row['CDL_MORNINGSTAR'] > 0 else 0
    state[51] = 1 if row['CDL_HAMMER'] > 0 else 0
    state[52] = 1 if row['CDL_SHOOTINGSTAR'] > 0 else 0
    state[53] = 1 if row['CDL_ENGULFING'] > 0 else 0
    state[54] = 1 if row['CDL_ENGULFING'] < 0 else 0

    if df['EMA50'].values[i] > df['EMA200'].values[i]:
        if df['EMA50'].values[i - 1] < df['EMA200'].values[i - 1]:
            state[55] = 1

    if df['EMA7'].values[i] > df['EMA50'].values[i]:
        if df['EMA7'].values[i - 1] < df['EMA50'].values[i - 1]:
            state[56] = 1

    delta = row['ATR'] / row['Close']

    state[57] = 1 if delta > 0.01 else 0
    state[58] = 1 if delta > 0.02 else 0
    state[59] = 1 if delta > 0.5 else 0

    # print('>>>>', i, len(dfHA))

    if dfHA['Open'].values[i] < dfHA['Close'].values[i]:
        state[60] = 1
    if dfHA['Open'].values[i-1] < dfHA['Close'].values[i-1]:
        state[61] = 1
    if dfHA['Open'].values[i-2] < dfHA['Close'].values[i-2]:
        state[62] = 1

    if state[60] == state[61] == state[62]:
        state[63] = 1

    if dfHA['Open'].values[i] > df['EMA21'].values[i] > dfHA['Close'].values[i]:
        state[64] = 1
    if dfHA['Open'].values[i] < df['EMA21'].values[i] < dfHA['Close'].values[i]:
        state[65] = 1

    if dfHA['Open'].values[i] > df['EMA50'].values[i] > dfHA['Close'].values[i]:
        state[66] = 1
    if dfHA['Open'].values[i] < df['EMA50'].values[i] < dfHA['Close'].values[i]:
        state[67] = 1

    if df['EMA21'].values[i] > df['EMA21'].values[i-1]:
        state[68] = 1
    if df['EMA50'].values[i] > df['EMA50'].values[i-1]:
        state[69] = 1
    if df['EMA7'].values[i] > df['EMA7'].values[i-1]:
        state[70] = 1
    if df['RSI'].values[i] > df['RSI'].values[i-1]:
        state[71] = 1

    if df['Low'].values[i] > df['Low'].values[i-1]:
        state[72] = 1
    if df['High'].values[i] > df['High'].values[i-1]:
        state[73] = 1
    if df['Low'].values[i] > df['Low'].values[i-2]:
        state[74] = 1
    if df['High'].values[i] > df['High'].values[i-2]:
        state[75] = 1

    return state
