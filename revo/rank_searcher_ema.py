
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


def run_backtest(ticker='AAPL', period='400d'):
    buy_signals = {}
    sell_signals = {}
    last_buy_position = 0
    macd_signal = 0

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval='1h')

    df.ta.sma(length=200, append=True, col_names=('SMA200',))
    df.ta.sma(length=50, append=True, col_names=('SMA50',))

    df.ta.supertrend(append=True, length=10, multiplier=4.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    df = df[200:].copy()

    purchase_price = 0

    stop_loss_price = 0
    take_profit_price = 0

    for i, (index, row) in enumerate(df.iterrows()):
        buy_signals[i] = False
        sell_signals[i] = False

        if last_buy_position == 0:
            if row['S_trend_d'] == 1 and df['S_trend_d'].values[i-1] < 0:
                if row['Close'] > row['SMA200']:
                    buy_signals[i] = True
                    last_buy_position = i
                    purchase_price = row['Close']

        if i > last_buy_position > 0:

            if row['S_trend_d'] < 0:
                sell_signals[i] = True
                last_buy_position = 0

    df['buy_signals'] = buy_signals.values()
    df['sell_signals'] = sell_signals.values()

    pf = vbt.Portfolio.from_signals(df.Close, entries=df['buy_signals'], exits=df['sell_signals'], freq='D',
                                    init_cash=10_000, fees=0.0025, slippage=0.0025)

    results = pf.stats()

    print(f'\n\n----\n {ticker}')
    print(results[['Start Value', 'End Value', 'Total Return [%]', 'Total Trades', 'Win Rate [%]']])

    return results['Total Return [%]']


if __name__ == '__main__':

    RANK = {}

    with open('smp500.txt', 'r') as f:
        TICKERS = f.readlines()

    TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
    TICKERS.remove('CEG')
    TICKERS.remove('ELV')

    # TICKERS = ['NVDA']

    for ticker in TICKERS[:10]:
        result = run_backtest(ticker, period='700d')
        RANK[ticker] = result

print(sorted(RANK, key=lambda x: RANK[x], reverse=True)[:10])
