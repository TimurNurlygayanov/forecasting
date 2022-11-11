# The example of script that do the backtesting of naive Bollinger Bands strategy:
# 1) Buy shares when price crosses Low Bollinger Bands line
# 2) Sell when you get > 5% profit from any deal
#
# With APPL shares for the last year it gives good results:
# total return: 23 %
# win rate 80%
#

import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


TAKE_PROFIT_THRESHOLD = 1.07  # 7 % of price increase


def run_backtest(ticker='AAPL', period='500d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)
    df.ta.supertrend(length=10, multiplier=4.0, append=True, col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    df = df[-300:]

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        previous_low = df['Low'].values[i-1]
        previous_L = df['L'].values[i-1]
        previous_close = df['Close'].values[i-1]
        previous_open = df['Close'].values[i - 1]

        if row['Low'] < row['L'] < row['Close'] and row['Open'] > row['Close']:
            buy_signals[i] = True
            last_buy_position = i

        # Sell as soon as we got desired profit:
        if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
            sell_signals[i] = True

        if row['Close'] > row['U']:
            sell_signals[i] = True

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])

    # Draw:
    graph = make_subplots(rows=1, cols=1)
    graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], row=1, col=1)


    graph.add_scatter(y=df['U'], mode='lines', name='Bollinger Upper',
                      line={'color': 'blue', 'width': 3})

    graph.add_scatter(y=df['L'], mode='lines', name='Bollinger Low',
                      line={'color': 'red', 'width': 3})

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15)

    graph.add_scatter(y=df['S_trend'], mode='lines', name='S_trend', row=1, col=1)
    graph.add_scatter(y=df['S_trend_s'], mode='lines', name='S_trend_s', row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#bada55', 'width': 3}, row=1, col=1)

    graph.show()


if __name__ == '__main__':

    for ticker in ['AAPL', 'MSFT', 'SEMR']:
        run_backtest(ticker)
