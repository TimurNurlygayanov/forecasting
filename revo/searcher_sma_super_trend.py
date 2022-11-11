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
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=1, cols=1)
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': 'magenta'})
    graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                      line={'color': 'blue', 'width': 3})

    graph.add_scatter(y=df['S_trend_d'], mode='lines', name='S_trend_d')
    graph.add_scatter(y=df['S_trend'], mode='lines', name='S_trend')
    graph.add_scatter(y=df['S_trend_s'], mode='lines', name='S_trend_s',
                      line={'color': '#ff4040', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#00ff7f', 'width': 3}, row=1, col=1)

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='400d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    # length=10, multiplier=4.0,
    df.ta.supertrend(append=True, length=34,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.cti(append=True, col_names=('CTI', ))

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        previous_close = df['Close'].values[i-1]
        previous_sma50 = df['SMA50'].values[i-1]
        previous_sma200 = df['SMA200'].values[i - 1]

        if row['Close'] > row['SMA50'] > row['SMA200'] > 0:
            if df['S_trend_d'].values[i-2] < 1 < df['S_trend_d'].values[i-1] + row['S_trend_d']:
                buy_signals[i] = True
                last_buy_position = i

        # Sell as soon as we got desired profit:
        if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
            sell_signals[i] = True

        # Sell when super trend finished
        if last_buy_position > 0:
            if row['S_trend_d'] < 0:
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

    for ticker in ['ALB', 'TSLA', 'HD', 'NEE', 'LLY']:
        run_backtest(ticker, period=f'{3 * 250}d')  # 1 year is 250 days
