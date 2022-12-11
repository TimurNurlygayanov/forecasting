# The example of script that do the backtesting of a naive strategy:
#
# Buy only if price and SMA50 are higher than SMA200:
#   - buy when the super trend started
#   - buy when SMA20 crosses SMA50
# Sell if:
#   - The profit is higher than TAKE_PROFIT_THRESHOLD
#   - The price crosses SMA20 from top to bottom and current profit > 5%
#   - SMA20 crosses SMA50 from top to bottom
#   - Super trend ended
#

import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots
# from scipy.signal import argrelextrema


RSI_PERIOD = 34
RSI_THRESHOLD = 40
TAKE_PROFIT_THRESHOLD = 1.30  # 30 % of price increase


def draw(df):
    graph = make_subplots(rows=1, cols=1)
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)

    # Crossover SMA20 & SMA50
    graph.add_scatter(y=df['SMA20'], mode='lines', name='SMA20',
                      line={'color': '#3399F5', 'width': 2}, row=1, col=1)
    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': '#FF9951', 'width': 3}, row=1, col=1)

    # The price and SMA50 should be higher than SMA200
    graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                      line={'color': 'blue', 'width': 3}, row=1, col=1)

    # Super Trend Up:
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l',
                      line={'color': '#00ff7f', 'width': 3}, row=1, col=1)

    # Buy signals
    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    # Sell signals
    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='600d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.supertrend(append=True, length=10, multiplier=4.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    df.ta.sma(length=20, append=True, col_names=('SMA20',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.sma(length=200, append=True, col_names=('SMA200',))

    # Cut first 200 days because there is no SMA200 data for this period
    df = df[200:].copy()

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[index] = False
        sell_signals[index] = False

        """
        # Search for local minimums:
        ilocs_min = argrelextrema(df['RSI'].values[:i], np.less_equal, order=20)[0]
        """

        if last_buy_position == 0:
            if row['Close'] > row['SMA200']:    # if price above 200 days moving average

                if row['S_trend_d'] > df['S_trend_d'].values[i-1]:   # and super trend just started
                    if row['Close'] < row['SMA50'] * 1.1:  # and price not larger than SMA50 more than 10%
                        buy_signals[index] = True
                        last_buy_position = i

                if row['SMA20'] > row['SMA50']:   # or SMA20 crossed SMA50 from the bottom to the top
                    if df['SMA20'].values[i-1] < df['SMA50'].values[i-1]:
                        if row['SMA200'] < row['SMA50'] < row['Close'] < row['SMA50'] * 1.1:   # and price higher then SMA50
                                                                               # but not larger than SMA50 more than 10%
                            buy_signals[index] = True
                            last_buy_position = i

        if i > last_buy_position > 0:
            if df['Close'].values[i] / df['Close'].values[last_buy_position] > TAKE_PROFIT_THRESHOLD:
                sell_signals[index] = True
                last_buy_position = 0

            if row['SMA20'] < row['SMA50']:
                if df['SMA20'].values[i-1] > df['SMA50'].values[i-1]:
                    sell_signals[index] = True
                    last_buy_position = 0

            if row['Close'] < row['SMA20']:
                if df['Close'].values[i] / df['Close'].values[last_buy_position] > 1.05:
                    sell_signals[index] = True
                    last_buy_position = 0

            if row['S_trend_d'] < 0:
                sell_signals[index] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)
    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    # 'ALB', 'TSLA', 'HD', 'NEE', 'MSFT', 'ALB', 'TSLA'
    for ticker in ['HD', 'TSLA', 'MSFT', 'NEE']:
        run_backtest(ticker)
