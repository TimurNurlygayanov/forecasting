# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#

import pandas as pd
import pandas_ta  # for TA magic
import ta.trend
import vectorbt as vbt
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import numpy as np


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.15  # 20 % of price increase
STOP_LOSS_THRESHOLD = 0.90  # stop loss on -10 % of price
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.7, 0.3])
    graph.update_layout(title=ticker)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['EMA5'], mode='lines', name='EMA10',
                      line={'color': 'magenta', 'width': 2})

    graph.add_scatter(y=df['EMA30'], mode='lines', name='EMA30',
                      line={'color': 'orange', 'width': 2})

    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': 'blue', 'width': 3})

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.add_scatter(y=df['volatility'], mode='lines', name='volatility',
                      line={'color': 'green', 'width': 2}, row=2, col=1)
    graph.add_scatter(y=df['volatility_long'], mode='lines', name='volatility_long',
                      line={'color': 'blue', 'width': 3}, row=2, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.ema(length=5, append=True, col_names=('EMA5',))
    df.ta.ema(length=30, append=True, col_names=('EMA30',))

    df.ta.sma(length=100, append=True, col_names=('SMA50',))

    volatility_data = [0]
    volatility_data_long = [0]

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        # Calculate volatility for the last N days
        volatility = df['EMA5'][:i].rolling(30).std(ddof=0)
        volatility_fast = df['EMA5'][:i].rolling(50).std(ddof=0)

        if i > 0:
            volatility_data.append(volatility[i-1])
            volatility_data_long.append(volatility_fast[i-1])

        # Search for local minimums:
        price_local_minimum = argrelextrema(df['EMA5'].values[:i], np.less_equal, order=30)[0]

        if last_buy_position == 0 and len(price_local_minimum) > 1:
            if i - price_local_minimum[-1] == 2 and volatility[i-1] > volatility[i-2]:
                if row['Close'] > df['Close'].values[i-1]:
                    buy_signals[i] = True
                    last_buy_position = i

        if i > last_buy_position > 0:
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

            if row['Close'] < STOP_LOSS_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()
    df['volatility'] = volatility_data
    df['volatility_long'] = volatility_data_long

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['ALB', 'TSLA', 'HD', 'NEE', 'NVDA']:
        run_backtest(ticker)
