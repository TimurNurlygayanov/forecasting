
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic

RANK = {}

def run_backtest(ticker='AAPL', period='700d'):

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval='1d')

    intervals = []

    df.ta.ema(length=200, append=True, col_names=('EMA200',))
    df.ta.ema(length=50, append=True, col_names=('EMA50',))
    df.ta.rsi(length=14, append=True, col_names=('RSI',))
    df.ta.atr(append=True, col_names=('ATR',))

    for i in range(3, 100, 5):
        df.ta.wma(length=i, append=True, col_names=(f'EMA{i}',))
        intervals.append(i)

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df = df[200:].copy()

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    total_profit = 1000
    max_total_profit = [1000, 0, 0, 0, 0]

    for i1 in intervals:
        for i2 in intervals:
            ema1 = f'EMA{i1}'
            ema2 = f'EMA{i2}'

            if ema1 == ema2:
                continue

            total_profit = 1000
            number_of_deals = 0
            good_deals = 0

            for i, (index, row) in enumerate(df.iterrows()):

                # crossover
                if purchase_price == 0:
                    if row[ema1] > row[ema2] and df[ema1].values[i-1] < df[ema2].values[i-1]:
                        purchase_price = row['Close']
                        # stop_loss_price = purchase_price * 0.97
                        # take_profit_price = purchase_price * 1.06
                        stop_loss_price = purchase_price - 2 * row['ATR']
                        take_profit_price = purchase_price + 6 * row['ATR']
                else:
                    if row['Low'] < stop_loss_price:
                        total_profit = (stop_loss_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1
                    elif row['High'] > take_profit_price:
                        # good deal:
                        total_profit = (take_profit_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1
                        good_deals += 1

            if max_total_profit[0] < total_profit:
                max_total_profit[0] = total_profit
                max_total_profit[1] = ema1
                max_total_profit[2] = ema2
                max_total_profit[3] = number_of_deals
                max_total_profit[4] = good_deals

        print(max_total_profit)

    print(f'Max result: {max_total_profit[0]}')
    print(f'EMA1: {max_total_profit[1]}')
    print(f'EMA2: {max_total_profit[2]}')
    print(f'number of total deals: {max_total_profit[3]}')
    print(f'number of good deals: {max_total_profit[4]}')


if __name__ == '__main__':

    ticker = 'F'
    run_backtest(ticker, period='700d')
