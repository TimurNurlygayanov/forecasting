# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#

import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import argrelextrema


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.15  # 15 % of price increase
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.2, 0.2])
    graph.update_layout(title=ticker)

    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )

    graph.update_layout(xaxis_rangeslider_visible=False)

    graph.add_candlestick(open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'])

    graph.add_bar(y=df['Volume'], name='Volume', row=2, col=1)

    graph.add_scatter(y=df['S_trend'], mode='lines', name='S_trend', row=1, col=1)
    graph.add_scatter(y=df['S_trend_s'], mode='lines', name='S_trend_s',
                      line={'color': '#ff4040', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#00ff7f', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_d'], mode='lines', name='S_trend_d', row=1, col=1)

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': '#8a2be2'}, row=1, col=1)
    graph.add_scatter(y=df['SMA21'], mode='lines', name='SMA21',
                      line={'color': '#ff4040'}, row=1, col=1)
    graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                      line={'color': '#cc0000', 'width': 4}, row=1, col=1)

    graph.add_scatter(y=df['CTI'], mode='lines', name='CTI',
                      line={'color': '#FF00DF'}, row=2, col=1)

    graph.add_scatter(y=df['VWMA_10'], mode='lines', name='VWMA_10',
                      line={'color': '#FF00DF'}, row=1, col=1)

    """
    graph.add_scatter(y=df['MACD'], mode='lines', name='MACD',
                      line={'color': 'black'}, row=3, col=1)
    graph.add_scatter(y=df['MACD_signal'], mode='lines', name='MACD_signal',
                      line={'color': 'red'}, row=3, col=1)
    """

    graph.show()


def run_backtest(ticker='AAPL', period='200d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    super_trend_finished = True

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.ema(length=21, append=True, col_names=('SMA21',))
    df.ta.ema(length=50, append=True, col_names=('SMA50',))
    df.ta.ema(length=200, append=True, col_names=('SMA200',))

    df.ta.cti(append=True, col_names=('CTI', ))

    df.ta.supertrend(length=10, multiplier=4.0, append=True, col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df.ta.vwma(append=True)

    # print(df.columns)
    # return 0

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        previous_cti = df['CTI'].values[i-1]
        previous_close = df['Close'].values[i-1]
        previous_super_trend_l = df['S_trend_l'].values[i-1]

        # if we do not keep position and the new super trend started, buy and hold
        if last_buy_position == 0 and super_trend_finished:
            if row['Close'] > row['SMA21'] > row['SMA50'] > row['SMA200'] and row['S_trend_l'] > 0:
                buy_signals[i] = True
                last_buy_position = i
                super_trend_finished = False

        if i > last_buy_position > 0:
            # if we get total % more than threshold - sell it
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

        # if super trend ended, sell it
        if np.isnan(row['S_trend_l']):
            if i > last_buy_position > 0:
                sell_signals[i] = True
                last_buy_position = 0

            super_trend_finished = True

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['AAPL', 'A', 'MSFT', 'NFLX']:
        run_backtest(ticker)
