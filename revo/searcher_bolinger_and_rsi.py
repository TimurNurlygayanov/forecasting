# The example of script that do the backtesting of combined strategy with:
#  - Bollinger Bands
#  - RSI
#
# Strategy (do not trust this strategy):
# 1) Buy when Low price crosses Low Bollinger Bands line and RSI < 33
# 2) Buy when RSI < 30
# 3) Sell when RSI > 70
# 4) Sell when price crosses Upper Bollinger Bands line with RSI > 65
# 5) Sell when we got the desired profit
#
# With APPL shares for the last year it gives good results:
# total return: 47 %
# win rate: 100%
#

import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
import plotly.graph_objects as go


RSI_PERIOD = 14
BBANDS_PERIOD = 30
TAKE_PROFIT_THRESHOLD = 1.10  # 10 % of price increase


def draw(df):
    graph = go.Figure()
    graph.update_layout(title=ticker)

    # Normalize all data to make it easy to read
    df['L'] /= df['Close'].max()
    df['U'] /= df['Close'].max()
    df['Close'] /= df['Close'].max()
    df['RSI'] /= 100.0

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3})

    graph.add_scatter(y=df['RSI'], mode='lines', name='RSI',
                      line={'color': 'grey'})

    graph.add_scatter(y=df['U'], mode='lines', name='Bollinger Upper',
                      line={'color': 'red'})

    graph.add_scatter(y=df['L'], mode='lines', name='Bollinger Low',
                      line={'color': 'red'})

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15)

    graph.show()


def run_backtest(ticker='AAPL', period='1y'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.rsi(length=RSI_PERIOD, append=True, col_names=('RSI',))
    df.ta.bbands(length=BBANDS_PERIOD, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if row['Close'] < row['L'] and row['RSI'] < 33:
            buy_signals[i] = True
            last_buy_position = i

        if row['RSI'] < 30:
            buy_signals[i] = True
            last_buy_position = i

        # Sell if share is overbought:
        if row['Close'] > row['U'] and row['RSI'] > 65:
            sell_signals[i] = True

        if row['RSI'] > 70:
            sell_signals[i] = True

        # Sell as soon as we got desired profit:
        if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'][last_buy_position]:
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

    for ticker in ['AAPL', 'MSFT', 'SEMR']:
        run_backtest(ticker)
