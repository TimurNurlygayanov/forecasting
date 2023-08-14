
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
from plotly.subplots import make_subplots

from bot.utils import get_data

RANK = {}
TAKE_PROFIT_THRESHOLD = 1.1  # 10 % of price increase
STOP_LOSSES_THRESHOLD = 0.9  # 10% risk


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    good_deals = 0
    bad_deals = 0
    total_sum = 1000

    df = get_data(ticker)

    if df is None:
        return total_sum

    df = df[200:].copy()

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0:
            """
            if row['EMA50'] > row['EMA200'] and df['EMA50'].values[i-1] < df['EMA200'].values[i-1]:
                buy_signals[i] = True
                last_buy_position = i
                purchase_price = row['Close']

                stop_loss_price = purchase_price * STOP_LOSSES_THRESHOLD
                take_profit_price = purchase_price * TAKE_PROFIT_THRESHOLD
            """
            if row['RSI'] > 30 > df['RSI'].values[i-1]:
                # if row['S_trend_d'] > 0 and row['S_trend_d_x'] > 0:
                #     if df['S_trend_d'].values[i-1] < 0 or df['S_trend_d_x'].values[i-1] < 0:
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

                total_sum *= TAKE_PROFIT_THRESHOLD

            if row['Low'] < stop_loss_price:
                sell_signals[i] = True
                last_buy_position = 0

                bad_deals += 1

                total_sum *= STOP_LOSSES_THRESHOLD

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    print(ticker, ' GOOD / BAD: ', good_deals, bad_deals, total_sum)

    return total_sum


if __name__ == '__main__':

    with open('smp500.txt', 'r') as f:
        TICKERS = f.readlines()

    TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
    TICKERS.remove('TSLA')

    for ticker in TICKERS[:100]:
        result = run_backtest(ticker, period='565d')
        RANK[ticker] = result

    print('Total:')
    print(sum(RANK.values()) / (1000*len(TICKERS[:100])))
