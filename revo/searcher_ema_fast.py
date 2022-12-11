# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.70  # 30 % of price increase
STOP_LOSSES_THRESHOLD = 0.80
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3})

    graph.add_scatter(y=df['EMA5'], mode='lines', name='EMA5')
    graph.add_scatter(y=df['EMA20'], mode='lines', name='EMA20')

    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': 'magenta'})
    graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                      line={'color': 'blue', 'width': 3})

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.add_scatter(y=df['T_index'], mode='lines', name='T_index', row=2, col=1)
    graph.add_scatter(y=df['T_index_signal'], mode='lines', name='T_index_signal',
                      row=2, col=1, line={'color': 'red'})

    graph.show()


def run_backtest(ticker='AAPL', period='600d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.ema(length=5, append=True, col_names=('EMA5',))
    df.ta.sma(length=20, append=True, col_names=('EMA20',))
    df = df[200:].copy()

    df['T_index'] = df['EMA20'] - df['SMA200']
    df['T_index_signal'] = df['EMA5'] - df['SMA200']

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0:
            if row['Open'] > row['SMA200']:
                if row['EMA20'] > row['SMA50'] and df['EMA20'].values[i-1] < df['SMA50'].values[i-1]:
                    buy_signals[i] = True
                    last_buy_position = i

        if i > last_buy_position > 0:
            # Sell as soon as we got total desired profit:
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

            if row['EMA5'] < row['EMA20']:
                sell_signals[i] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=10_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['ALB', 'TSLA', 'A', 'NVDA']:
        run_backtest(ticker, period=f'{3 * 250}d')  # 1 year is 250 days
