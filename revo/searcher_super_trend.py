# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
# import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.70  # 30 % of price increase
STOP_LOSSES_THRESHOLD = 0.85
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=1, cols=1)
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    x = None # df.index

    # graph.add_candlestick(x=x, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    graph.add_scatter(y=df['Close'], x=x, mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)


    graph.add_scatter(y=df['SMA200'], x=x, mode='lines', name='SMA200',
                      line={'color': 'blue', 'width': 3})
    graph.add_scatter(y=df['EMA10'], x=x, mode='lines', name='EMA10',
                      line={'color': 'orange', 'width': 2})

    # graph.add_scatter(y=df['S_trend_d'], mode='lines', name='S_trend_d')
    # graph.add_scatter(y=df['S_trend'], mode='lines', name='S_trend')
    # graph.add_scatter(y=df['S_trend_s'], x=df.index, mode='lines', name='S_trend_s',
    #                   line={'color': '#ff4040', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], x=x, mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#00ff7f', 'width': 3}, row=1, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='600d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    macd_signal = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval="1h")

    # length=10, multiplier=4.0,
    df.ta.supertrend(append=True, length=10, multiplier=4.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.wma(length=34, append=True, col_names=('EMA10',))

    df = df[500:].copy()

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if i > last_buy_position > 0:
            # Sell if super trend already finished
            if row['S_trend_d'] < 0:
                sell_signals[i] = True
                last_buy_position = 0

            # Sell as soon as we got total desired profit:
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

            # Stop loses at 8% of loses:
            if row['Close'] < STOP_LOSSES_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    draw(df)

    print(f'\n\n----\n {ticker}')


if __name__ == '__main__':

    for ticker in ['TSLA']:   # 'ALB', 'TSLA', 'NVDA'
        run_backtest(ticker, period=f'{700}d')  # 1 year is 250 days
