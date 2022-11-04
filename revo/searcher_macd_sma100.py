# strategy from https://www.youtube.com/watch?v=o_SUdccjuC4
#

import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 14
BBANDS_PERIOD = 34
RSI_THRESHOLD = 33
TAKE_PROFIT_THRESHOLD = 1.30  # 30 % of price increase
MAX_LENGTH_FOR_BET = 20


def draw(df):
    # graph = go.Figure()
    graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])
    graph.update_layout(title=ticker)

    # Normalize all data to make it easy to read
    df['U'] /= df['Close'].max()
    df['L'] /= df['Close'].max()

    df['EMA_short'] /= df['Close'].max()
    df['EMA_long'] /= df['Close'].max()
    df['SMA_long'] /= df['Close'].max()
    df['Close'] /= df['Close'].max()
    df['RSI'] /= 100.0

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

    # df.ta.rsi(length=RSI_PERIOD, append=True, col_names=('RSI', ))
    # df.ta.bbands(length=BBANDS_PERIOD, std=2.3, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

    df.ta.ema(length=10, append=True, col_names=('EMA_short',))
    df.ta.ema(length=21, append=True, col_names=('EMA_long',))
    df.ta.sma(length=100, append=True, col_names=('SMA_long',))

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    # df.ta.adx(append=True, col_names=('ADX', 'DMP', 'DMN'))

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        previous_macd = df['MACD'].values[i - 1]

        if row['Close'] > row['EMA_long'] > row['SMA_long'] > 0:
            if 0 > row['MACD'] > row['MACD_signal'] > previous_macd:
                if abs(row['Close'] - row['SMA_long']) / row['SMA_long'] < 0.1:
                    buy_signals[i] = True
                    last_buy_position = i

        if row['MACD'] > 5:
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
