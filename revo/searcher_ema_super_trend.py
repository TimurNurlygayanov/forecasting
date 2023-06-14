# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 10
TAKE_PROFIT_THRESHOLD = 1.01
STOP_LOSSES_THRESHOLD = 0.85


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=1, cols=1, vertical_spacing=0.01)
    graph.update_layout(title=ticker)

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

    graph.add_scatter(y=df['EMA200'], mode='lines', name='EMA200',
                      line={'color': 'blue', 'width': 3})

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
    df = df.ta.ticker(ticker, period='700d', interval='1h')

    # length=10, multiplier=4.0,
    df.ta.supertrend(append=True, length=14, multiplier=2.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))

    df.ta.ema(length=200, append=True, col_names=('EMA200',))
    purchase_price = 0
    trend_finished = 1

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0 and trend_finished:
            if row['Close'] > row['EMA200']:
                if row['S_trend_d'] > 0:
                    buy_signals[i] = True
                    last_buy_position = i
                    purchase_price = row['Close']

                    trend_finished = False

        if row['S_trend_d'] < 0:
            trend_finished = 1

        if i > last_buy_position > 0:

            if row['Low'] < row['S_trend_l'] or row['S_trend_d'] < 0:
                sell_signals[i] = True
                last_buy_position = 0

            if row['Close'] > TAKE_PROFIT_THRESHOLD * purchase_price:
                sell_signals[i] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='H',
                                    init_cash=10_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['NVDA', 'MSFT']:
        run_backtest(ticker, period=f'{3 * 250}d')  # 1 year is 250 days
