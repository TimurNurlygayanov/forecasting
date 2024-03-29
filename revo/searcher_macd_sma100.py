# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#

import pandas as pd
import pandas_ta  # for TA magic
import ta.trend
import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.10  # 30 % of price increase
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])
    graph.update_layout(title=ticker)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['EMA_short'], mode='lines', name='EMA_short',
                      line={'color': 'magenta'}, row=1, col=1)
    graph.add_scatter(y=df['EMA_long'], mode='lines', name='EMA_long',
                      line={'color': 'orange', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['SMA_long'], mode='lines', name='SMA_long',
                      line={'color': 'blue', 'width': 3}, row=1, col=1)

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


def run_backtest(ticker='AAPL', period='700d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.ema(length=10, append=True, col_names=('EMA_short',))
    df.ta.ema(length=21, append=True, col_names=('EMA_long',))
    df.ta.sma(length=100, append=True, col_names=('SMA_long',))
    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df = df[100:].copy()

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        df['MACD'] = ta.trend.macd(close=df[:i]['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(close=df[:i]['Close'])

        # Make sure MACD is normalized, all values are from -1 to 1
        df['MACD_signal'] /= df['MACD'].max()
        df['MACD'] /= df['MACD'].max()

        volList = df['MACD'].rolling(20).std(ddof=0)

        if last_buy_position == 0:
            if row['Close'] > row['EMA_long'] > row['SMA_long'] > 0:
                if 0.1 > df['MACD'].values[i-1] > df['MACD_signal'].values[i-1] and df['MACD'].values[i-2] < df['MACD_signal'].values[i-2]:
                    if volList[i-1] > 0.10:
                        buy_signals[i] = True
                        last_buy_position = i

        if i > last_buy_position > 0:
            if df['MACD'].values[i-1] > 0.9 and row['EMA_short'] > row['Close']:
                sell_signals[i] = True
                last_buy_position = 0

            if row['EMA_short'] > row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True
                last_buy_position = 0

            """
            if row['EMA_long'] > row['EMA_short']:
                sell_signals[i] = True
                last_buy_position = 0
            """

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['ALB', 'TSLA', 'HD', 'NEE', 'NVDA']:
        run_backtest(ticker)
