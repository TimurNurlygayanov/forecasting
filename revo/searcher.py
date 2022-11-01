# The example of script that do the backtesting on simple RSI strategy:
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


RSI_PERIOD = 14
RSI_TRESHOLD = 35
TAKE_PROFIT_TRESHOLD = 1.05  # 7 % of price increase


def run_backtest(ticker='AAPL', period='1y'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    df.ta.rsi(length=RSI_PERIOD, append=True)

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[index] = False
        sell_signals[index] = False

        previous_rsi_value = df[f'RSI_{RSI_PERIOD}'].values[i-1]
        rsi_value = df[f'RSI_{RSI_PERIOD}'].values[i]

        if rsi_value < RSI_TRESHOLD < previous_rsi_value:
            buy_signals[index] = True
            last_buy_position = i

        # Sell as soon as we got desired profit:
        if df['Close'].values[i] > TAKE_PROFIT_TRESHOLD * df['Close'].values[last_buy_position]:
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
