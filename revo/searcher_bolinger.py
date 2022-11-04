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


TAKE_PROFIT_THRESHOLD = 1.07  # 7 % of price increase
BBANDS_PERIOD = 34


def run_backtest(ticker='AAPL', period='1y'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.bbands(length=BBANDS_PERIOD, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[index] = False
        sell_signals[index] = False

        current_price = df['Close'].values[i]
        current_lower_bollinger_threshold = df['L'].values[i]

        if current_price < current_lower_bollinger_threshold:
            buy_signals[index] = True
            last_buy_position = i

        # Sell as soon as we got desired profit:
        if df['Close'].values[i] > TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
            sell_signals[index] = True

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=3_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])


if __name__ == '__main__':

    for ticker in ['AAPL', 'MSFT', 'SEMR']:
        run_backtest(ticker)
