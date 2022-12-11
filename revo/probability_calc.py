import numpy as np
import pandas as pd
import pandas_ta  # for TA magic


PERIOD = 5         # 20 days
MIN_PROFIT = 1.05  # 1 %


def run_backtest(ticker='AAPL', period='3y'):
    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period)

    got_profit = {}

    for i, (index, row) in enumerate(df.iterrows()):
        got_profit[index] = False

        if df['Close'].values[i:i+PERIOD].max() / row['Close'] > MIN_PROFIT:
            got_profit[index] = True

    print(f'{ticker} {100 * list(got_profit.values()).count(True) / len(got_profit):.1f}% probability to get profit')


if __name__ == '__main__':

    # 'ALB', 'TSLA', 'HD', 'NEE', 'MSFT', 'NVDA'
    for ticker in ['ALB', 'TSLA', 'HD', 'NEE', 'MSFT', 'NVDA']:
        run_backtest(ticker)
