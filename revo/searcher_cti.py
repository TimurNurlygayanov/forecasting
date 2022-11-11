# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#

import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.20  # 30 % of price increase
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])
    graph.update_layout(title=ticker)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['S_trend'], mode='lines', name='S_trend', row=1, col=1)
    graph.add_scatter(y=df['S_trend_s'], mode='lines', name='S_trend_s', row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#bada55', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_d'], mode='lines', name='S_trend_d', row=1, col=1)

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': '#FF00DF'}, row=1, col=1)
    graph.add_scatter(y=df['SMA10'], mode='lines', name='SMA10',
                      line={'color': '#339955'}, row=1, col=1)

    graph.add_scatter(y=df['CTI'], mode='lines', name='CTI',
                      line={'color': '#FF00DF'}, row=2, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}
    cti_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.ema(length=10, append=True, col_names=('SMA10',))
    df.ta.ema(length=50, append=True, col_names=('SMA50',))

    df.ta.cti(append=True, col_names=('CTI', ))

    df.ta.supertrend(length=10, multiplier=4.0, append=True, col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False
        cti_signals[i] = False

        previous_close = df['Close'].values[i-1]

        # Search for local minimums:
        ilocs_min = argrelextrema(df['CTI'].values[:i+1], np.less_equal, order=10)[0]

        if len(ilocs_min) >= 1:
            if i - ilocs_min[-1] == 1 and row['CTI'] < -0.5:
                cti_signals[ilocs_min[-1]] = True

                if row['Close'] > previous_close:

                    buy_signals[i] = True
                    last_buy_position = i

        if last_buy_position > 0:
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['AAPL', 'A', 'MSFT']:
        run_backtest(ticker)
