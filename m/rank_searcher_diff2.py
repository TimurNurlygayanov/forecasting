
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic

RANK = {}

def run_backtest(ticker='AAPL', period='700d'):

    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval='1d')

    intervals = []
    trends = []

    df.ta.atr(append=True, col_names=('ATR',))

    for i in [9, 21, 34, 50, 100, 150, 200]:
        df.ta.wma(length=i, append=True, col_names=(f'WMA{i}',))
        intervals.append(i)

    for i in [10, 12, 21, 34]:
        for j in [2, 3, 4, 6]:
            df.ta.supertrend(append=True, length=i, multiplier=j,
                             col_names=(f'S_trend_{i}_{j}', f'S_trend_d_{i}_{j}', f'S_trend_l_{i}_{j}', f'S_trend_s_{i}_{j}',))
            trends.append(f'S_trend_d_{i}_{j}')

    df = df[200:].copy()

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    max_total_profit = [1000, 0, 0, 0, 0, 1000]

    for i1 in intervals:
        for t2 in trends:
            ema1 = f'WMA{i1}'

            total_profit = 1000
            number_of_deals = 0
            good_deals = 0
            purchase_index = 0
            average_period = 0

            for i, (index, row) in enumerate(df.iterrows()):

                # crossover
                if purchase_price == 0:
                    if row['Close'] > row[ema1] and df[t2].values[i-1] < 0 < row[t2]:
                        purchase_price = row['Close']
                        # stop_loss_price = purchase_price * 0.94
                        # take_profit_price = purchase_price * 1.21
                        stop_loss_price = purchase_price - 2 * row['ATR']
                        take_profit_price = purchase_price + 6 * row['ATR']

                        purchase_index = i

                        # print('loss', 100 * (2 * row['ATR'] / purchase_price))
                        # print('profit', 100 * (6 * row['ATR'] / purchase_price))
                else:
                    if row['Low'] < stop_loss_price:
                        total_profit = (stop_loss_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1

                        average_period += i - purchase_index
                    elif row['High'] > take_profit_price:
                        # good deal:
                        total_profit = (take_profit_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1
                        good_deals += 1

                        average_period += i - purchase_index

            average_period = average_period / number_of_deals if number_of_deals > 0 else 1
            if max_total_profit[0] < total_profit:
                max_total_profit[0] = total_profit
                max_total_profit[1] = ema1
                max_total_profit[2] = t2
                max_total_profit[3] = number_of_deals
                max_total_profit[4] = good_deals
                max_total_profit[5] = average_period

        print(max_total_profit)

    print(f'Max result: {max_total_profit[0]}')
    print(f'WMA1: {max_total_profit[1]}')
    print(f'Super trend: {max_total_profit[2]}')
    print(f'number of total deals: {max_total_profit[3]}')
    print(f'number of good deals: {max_total_profit[4]}')
    print(f'average period of hold: {max_total_profit[5]}')


if __name__ == '__main__':

    ticker = 'NVDA'
    run_backtest(ticker, period='500d')
