# The example of script that do the backtesting of naive RSI strategy:
# 1) Buy shares when RSI crosses 35 from top to down
# 2) Sell when you get > 5% profit from any deal
#
# With APPL shares for the last year it gives good results:
# total return: 23 %
# win rate 100%
#
# useful videos:
#  https://www.youtube.com/watch?v=W_kKPp9LEFY
#  https://www.youtube.com/watch?v=57hsQz70vVE
#

import pandas as pd
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


RSI_PERIOD = 14
RSI_THRESHOLD = 35
TAKE_PROFIT_THRESHOLD = 1.10  # 5 % of price increase


def draw(df):
    graph = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.2, 0.2])
    graph.update_layout(title=ticker)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.add_scatter(y=df['RSI'], mode='lines', name='RSI', row=2, col=1)

    graph.add_scatter(y=df['volatility'], mode='lines', name='volatility',
                      line={'color': 'green', 'width': 2}, row=3, col=1)

    graph.show()


def run_backtest(ticker='AAPL', period='1y'):
    buy_signals = {}
    sell_signals = {}
    volatility_data = [0]
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.rsi(length=RSI_PERIOD, append=True, col_names=('RSI', ))
    df.ta.ema(length=5, append=True, col_names=('EMA5',))

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[index] = False
        sell_signals[index] = False

        previous_rsi_value = df['RSI'].values[i-1]

        volatility = df['EMA5'][:i].rolling(30).std(ddof=0)

        if i > 0:
            volatility_data.append(volatility[i - 1])

        if last_buy_position == 0:
            # If RSI lower than threshold and volatility is moving down...
            if row['RSI'] < RSI_THRESHOLD < previous_rsi_value:
                if volatility[i - 1] < volatility[i - 2]:
                    buy_signals[index] = True
                    last_buy_position = i

        if i > last_buy_position > 0:
            if row['RSI'] > 70:
                sell_signals[index] = True
                last_buy_position = 0

            # Sell as soon as we got desired profit:
            if row['Close'] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[index] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()
    df['volatility'] = volatility_data

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    draw(df)

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['AAPL', 'MSFT', 'SEMR']:
        run_backtest(ticker)
