
import random

from polygon import RESTClient
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pandas_ta  # for TA magic


now = datetime.now()
START = now - timedelta(days=1000)
END = now + timedelta(days=1)

with open('/Users/timur.nurlygaianov/api_key.txt', encoding='utf-8', mode='r') as f:
    api_key = f.readlines()
    api_key = ''.join(api_key).strip()

client = RESTClient(api_key=api_key)


def get_data(ticker='AAPL', forex=False):
    df = None
    indexes = []
    data = {'Close': [], 'Open': [], 'Low': [], 'High': [], 'vwap': [], 'volume': []}

    try:
        for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day",  # "hour"
                                  from_=START.strftime("%Y-%m-%d"),
                                  to=END.strftime("%Y-%m-%d"),
                                  limit=50000):
            date = datetime.fromtimestamp(a.timestamp // 1000).strftime("%Y-%m-%d %H:%M:%S")

            indexes.append(date)
            data['Close'].append(a.close)
            data['Open'].append(a.open)
            data['High'].append(a.high)
            data['Low'].append(a.low)
            data['vwap'].append(a.vwap)
            data['volume'].append(a.volume)

        df = pd.DataFrame(data, index=indexes)

        df.ta.ema(length=50, append=True, col_names=('EMA50',))
        df.ta.ema(length=200, append=True, col_names=('EMA200',))
        df.ta.supertrend(append=True, length=10, multiplier=3.0,
                         col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
        df.ta.rsi(length=14, append=True, col_names=('RSI',))
        df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))
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


def get_tickets():
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
