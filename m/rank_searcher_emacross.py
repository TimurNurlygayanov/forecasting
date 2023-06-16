
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RANK = {}
TAKE_PROFIT_THRESHOLD = 1.05  # 4 % of price increase
STOP_LOSSES_THRESHOLD = 0.91  # 2% risk


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    good_deals = 0
    bad_deals = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval='1h')

    df.ta.ema(length=21, append=True, col_names=('EMA21',))
    df.ta.ema(length=5, append=True, col_names=('EMA5',))
    # df.ta.rsi(length=14, append=True, col_names=('RSI',))

    # df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df = df[200:].copy()

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0:
            if row['EMA5'] > row['EMA21'] and df['EMA5'].values[i-1] < df['EMA21'].values[i-1]:
                buy_signals[i] = True
                last_buy_position = i
                purchase_price = row['Close']

                stop_loss_price = purchase_price * STOP_LOSSES_THRESHOLD
                take_profit_price = purchase_price * TAKE_PROFIT_THRESHOLD

        if i > last_buy_position > 0:

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

    print(ticker, ' GOOD / BAD: ', good_deals, bad_deals)

    K = 0 if not bad_deals else good_deals / bad_deals

    return K


if __name__ == '__main__':

    with open('smp500.txt', 'r') as f:
        TICKERS = f.readlines()

    TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
    TICKERS.remove('CEG')
    TICKERS.remove('ELV')
    TICKERS.remove('TSLA')

    for ticker in TICKERS[:20]:
        result = run_backtest(ticker, period='700d')
        RANK[ticker] = result


print(sorted(RANK, key=lambda x: RANK[x], reverse=True)[:10])
