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
TAKE_PROFIT_THRESHOLD = 1.10  # 30 % of price increase
STOP_LOSSES_THRESHOLD = 0.95
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    # graph.add_scatter(y=df['Close'], mode='lines', name='Price',
    #                   line={'color': 'green', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['EMA9'], mode='lines', name='EMA9')
    graph.add_scatter(y=df['EMA21'], mode='lines', name='EMA21')

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
    df = df.ta.ticker(ticker, period=period, interval='1h')

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.ema(length=9, append=True, col_names=('EMA9',))
    df.ta.ema(length=14, append=True, col_names=('EMA21',))

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df = df[200:].copy()

    purchase_price = 0

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0:
            if row['EMA9'] > row['EMA21']:
                if df['EMA9'].values[i-1] < df['EMA21'].values[i-1] < df['Close'].values[i-1]:
                    buy_signals[i] = True
                    last_buy_position = i
                    purchase_price = row['Close']

        if i > last_buy_position > 0:
            """
            if row['High'] > TAKE_PROFIT_THRESHOLD * purchase_price:
                sell_signals[i] = True
                last_buy_position = 0
            """
            if row['Close'] < row['EMA21']:
                sell_signals[i] = True
                last_buy_position = 0

            if row['Low'] < STOP_LOSSES_THRESHOLD * purchase_price:
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

    for ticker in ['ALB', 'TSLA', 'HD', 'NEE', 'NVDA']:
        run_backtest(ticker, period=f'700d')  # 1 year is 250 days
