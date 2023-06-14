
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RANK = {}
TAKE_PROFIT_THRESHOLD = 1.04  # 6 % of price increase
STOP_LOSSES_THRESHOLD = 0.98  # 2% risk


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    good_deals = 0
    bad_deals = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval='1h')

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.wma(close=df['Low'], length=5, offset=5, append=True, col_names=('WMA3',))

    df.ta.supertrend(append=True, length=10, multiplier=3.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    df = df[200:].copy()

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0:
            if row['S_trend_d'] < 0:
                if row['Low'] > row['WMA3'] and df['Low'].values[i-1] < df['WMA3'].values[i-1]:
                    buy_signals[i] = True
                    last_buy_position = i
                    purchase_price = row['Close']

                    stop_loss_price = purchase_price * STOP_LOSSES_THRESHOLD
                    take_profit_price = purchase_price * TAKE_PROFIT_THRESHOLD

        if i > last_buy_position > 0:

            if row['S_trend_d'] < 0:
                sell_signals[i] = True
                last_buy_position = 0

            if row['High'] > take_profit_price:
                sell_signals[i] = True
                last_buy_position = 0

                good_deals += 1

            if row['Low'] < stop_loss_price:
                sell_signals[i] = True
                last_buy_position = 0

                bad_deals += 1

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    print('GOOD / BAD: ', good_deals, bad_deals)

    K = 0 if not bad_deals else good_deals / bad_deals

    return K


if __name__ == '__main__':

    with open('smp500.txt', 'r') as f:
        TICKERS = f.readlines()

    TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
    TICKERS.remove('CEG')
    TICKERS.remove('ELV')

    for ticker in TICKERS[:10]:
        result = run_backtest(ticker, period='700d')
        RANK[ticker] = result


print(sorted(RANK, key=lambda x: RANK[x], reverse=True)[:10])
