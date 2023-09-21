
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
START = now - timedelta(days=1500)
END = now + timedelta(days=1)

with open('/Users/timur.nurlygaianov/api_key.txt', encoding='utf-8', mode='r') as f:
    api_key = f.readlines()
    api_key = ''.join(api_key).strip()

client = RESTClient(api_key=api_key)


def get_ticker_details(ticker='AAPL'):
    result = client.get_ticker_details(ticker=ticker)
    # print(result)
    return result


def get_tickers_polygon(limit=1000):
    result = set()

    file_name = f'cached_tickers_{limit}.txt'
    if os.path.isfile(file_name):
        with open(file_name, 'r') as f:
            result = set(f.readlines())
            result = set([r.strip() for r in result])
            return sorted(list(result)[:limit])

    # print('Collecting list of exchanges...')
    # exchanges = pd.DataFrame(client.get_exchanges(asset_class='stocks', locale='us'))
    # exchanges = set(exchanges.mic)
    # exchanges.remove(None)
    # print(f'Identified {len(exchanges)} exchanges.')

    print('Collecting list of tickers...')

    for e in ['XNAS', 'XNYS']:   # NASDAQ and NYSE only
        for x in client.list_tickers(market='stocks', exchange=e, active=True, limit=1000, type='CS'):
            if '.' not in x.ticker and x.ticker == x.ticker.upper():
                ticker_data = client.get_ticker_details(ticker=x.ticker)

                if ticker_data.list_date and ticker_data.list_date < '2021-01-01':   # IPO date is more than 2 years ago
                    # if ticker_data.market_cap and ticker_data.market_cap > 300 * (10 ** 2):   # Market Cap > 300M
                    #     if ticker_data.weighted_shares_outstanding and ticker_data.weighted_shares_outstanding
                    #     > 10 ** 6:   # Shares outstanding over 1M

                    if len(result) >= limit:
                        break
                    else:
                        result.add(x.ticker)

    print(f'Collected {len(result)} tickers.')

    with open(file_name, 'w+') as f:
        f.writelines('\n'.join(result))

    return list(result)[:limit]


def get_news(ticker):
    url = (f'https://api.polygon.io/v2/reference/news?ticker={ticker}&'
           f'order=desc&limit=1000&sort=published_utc&apiKey={api_key}')

    results = []
    while url:
        res = requests.get(url).json()

        for n in res['results']:
            date = datetime.strptime(n['published_utc'].split('T')[0], "%Y-%m-%d")

            results.append({
                'date': date,
                'keywords': n.get('keywords'),
                'tickers': n.get('tickers'),
                'title': n['title']
            })

            if date < START:
                return results

        url = res.get('next_url')

        if url:
            url += f'&apiKey={api_key}'

    return results


def get_data(ticker='AAPL', period='day'):
    df = None
    indexes = []
    data = {'Close': [], 'Open': [], 'Low': [], 'High': [], 'vwap': [], 'volume': []}

    try:
        file_name = f'rl/data/{ticker}_daily_{datetime.now().strftime("%Y-%m-%d")}.xlsx'
        if os.path.isfile(file_name):
            df = pd.read_excel(file_name, index_col=0)
        else:
            for a in client.list_aggs(ticker=ticker, multiplier=1, timespan=period,  # "hour"
                                      from_=START.strftime("%Y-%m-%d"),
                                      to=END.strftime("%Y-%m-%d"),
                                      limit=50000):
                date = datetime.fromtimestamp(a.timestamp // 1000).strftime("%Y-%m-%d")

                indexes.append(date)
                data['Close'].append(a.close)
                data['Open'].append(a.open)
                data['High'].append(a.high)
                data['Low'].append(a.low)
                data['vwap'].append(a.vwap)
                data['volume'].append(a.volume)

            df = pd.DataFrame(data, index=indexes)
            df.to_excel(file_name, index=True, header=True)

        """
        df.ta.cdl_pattern(append=True, name=["doji", "morningstar", "hammer", "engulfing", "shootingstar"])

        df.ta.ema(length=7, append=True, col_names=('EMA7',))
        df.ta.ema(length=50, append=True, col_names=('EMA50',))
        df.ta.ema(length=200, append=True, col_names=('EMA200',))
        df.ta.supertrend(append=True, length=10, multiplier=3.0,
                         col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
        df.ta.supertrend(append=True, length=34, multiplier=4.0,
                         col_names=('S_trend34', 'S_trend_d34', 'S_trend_l34', 'S_trend_s34',))
        df.ta.rsi(length=14, append=True, col_names=('RSI',))
        df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))
        df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)

        df.ta.atr(append=True, col_names=('ATR',))
        """
    except Exception as e:
        print(f'No data for {ticker} {e}')

    return df


def check_strategy(df, rsi_threshold=50):
    good_deals = 0
    bad_deals = 0
    deal_length = []

    for i, (index, row) in enumerate(df.iterrows()):
        if row['RSI'] > rsi_threshold > df['RSI'].values[i-1]:
            buy_price = row['Close']
            stop_loss = buy_price * 0.93
            take_profit = buy_price * 1.1

            for j in range(i+1, i+100):
                if j < df.__len__():
                    if df['Low'].values[j] < stop_loss:
                        bad_deals += 1
                        deal_length.append(j - i)
                        break
                    if df['High'].values[j] > take_profit:
                        good_deals += 1
                        deal_length.append(j - i)
                        break

    if bad_deals + good_deals == 0:
        return 0, 0

    return good_deals / (good_deals + bad_deals), sum(deal_length) / len(deal_length)


def get_tickers():
    with open('smp500.txt', 'r') as f:
        TICKERS = f.readlines()

    TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
    TICKERS.remove('TSLA')
    TICKERS.remove('ABMD')
    TICKERS.remove('ANTM')
    TICKERS.remove('BLL')
    TICKERS.remove('CERN')
    TICKERS.remove('CTXS')
    TICKERS.remove('FRC')
    TICKERS.remove('FB')
    TICKERS.remove('DRE')
    TICKERS.remove('DISCK')
    TICKERS.remove('DISCA')
    TICKERS.remove('FISV')
    TICKERS.remove('TWTR')
    TICKERS.remove('FBHS')
    TICKERS.remove('VIAC')
    TICKERS.remove('KSU')
    TICKERS.remove('NLSN')
    TICKERS.remove('SIVB')
    TICKERS.remove('PBCT')
    TICKERS.remove('XLNX')
    TICKERS.remove('PKI')
    TICKERS.remove('INFO')
    TICKERS.remove('WLTW')
    TICKERS.remove('NLOK')
    TICKERS.remove('KHC')

    random.shuffle(TICKERS)

    return TICKERS


def is_engulfing_bullish(open1, open2, close1, close2):
    if open1 > close1:   # previous candle is red
        if open2 < close2:  # current candle is green
            if open2 < close1 and close2 > open1:  # green is bigger
                return 1

    return 0


def get_state(df, i: int = 0, step_size: int = 10):
    state = [0] * 64
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

    state[38] = 1 if df['vwap'].values[i] > df['vwap'].values[i - 1] else 0
    state[39] = 1 if df['vwap'].values[i - 1] > df['vwap'].values[i - 2] else 0
    state[40] = 1 if df['vwap'].values[i - 2] > df['vwap'].values[i - 3] else 0

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

    state[60] = 1 if row['News'] else 0
    state[61] = 1 if sum(df['News'].values[i - 5:i]) > 0 else 0
    state[62] = 1 if sum(df['News'].values[i - 5:i]) > 2 else 0
    state[63] = 1 if sum(df['News'].values[i - 10:i]) > 0 else 0

    return state


def get_features_importance(model):
    # Get feature importances
    feature_importance = model.get_feature_importance()

    # Get feature names
    feature_names = model.feature_names_

    # Create a dictionary to associate feature names with importances
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Sort features by importance
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print sorted feature importances
    for feature, importance in sorted_feature_importance[:10]:
        print(f"{feature}: {importance:.4f}")
