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

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

    graph.add_scatter(y=df['WMA10'], mode='lines', name='WMA10',
                      line={'color': 'orange'})
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

    graph.add_scatter(y=df['MACD'], mode='lines', name='MACD',
                      line={'color': 'black'}, row=2, col=1)
    graph.add_scatter(y=df['MACD_signal'], mode='lines', name='MACD_signal',
                      line={'color': 'red'}, row=2, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='400d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    macd_signal = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    # length=10, multiplier=4.0,
    df.ta.supertrend(append=True, length=10, multiplier=4.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))
    df.ta.wma(length=10, append=True, col_names=('WMA10',))

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df.ta.bbands(length=BBANDS_PERIOD, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        last_sma50 = df['SMA50'].values[i-1]
        last_wma10 = df['WMA10'].values[i-1]

        if last_buy_position == 0:
            if row['Close'] > row['WMA10'] > row['SMA50'] > row['SMA200'] and row['S_trend_d'] > 0:
                if row['MACD'] > row['MACD_signal']:
                    buy_signals[i] = True
                    last_buy_position = i

        if i > last_buy_position > 0:
            """
            # Sell if super trend already finished
            if row['S_trend_d'] < 0:
                sell_signals[i] = True
                last_buy_position = 0
            """

            """
            # Sell as soon as we got total desired profit:
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

            # Stop loses at 8% of loses:
            if row['WMA10'] < row['EMA100']:
                sell_signals[i] = True
                last_buy_position = 0
            """

            if len(df['MACD'].dropna().values) > 20:
                max_macd = df['MACD'].values[i-20:i].max()
                if row['MACD'] / max_macd > 0.7 and row['MACD'] < row['MACD_signal']:
                    sell_signals[i] = True
                    last_buy_position = 0

            """
            # Sell if we got more than 2 % daily rate
            current_profit = 1 - row['Close'] / df['Close'].values[last_buy_position]
            if current_profit / (i - last_buy_position) > 0.5:
                sell_signals[i] = True
                last_buy_position = 0
            """


    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=10_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['ALB', 'TSLA', 'HD', 'NEE', 'NVDA']:
        run_backtest(ticker, period=f'{3 * 250}d')  # 1 year is 250 days
