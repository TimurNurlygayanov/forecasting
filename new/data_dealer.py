# !/usr/bin/python3
# -*- encoding=utf8 -*-

from catboost import Pool
from catboost import CatBoostClassifier

import pandas as pd

from os import path
import yfinance as yf


class Data:

    tickers = {}
    short_period = 5
    long_period = 25
    short_benefit = 1.05  # 5% raise in next short_period
    long_benefit = 1.10   # 10% raise in next long_period

    def __init__(self):
        """ Get all data and convert it to the right way. """

        with open('revolut_tickers.txt', 'r') as f:
            TICKERS = f.readlines()

        TICKERS = [t.replace('\n', '') for t in TICKERS]
        TICKERS = TICKERS[:100]

        if path.exists('cached_data.xlsx'):
            # Load the data from file, if the file exists
            data = pd.read_excel('cached_data.xlsx', index_col=0, header=[0, 1])
        else:
            data = yf.download(' '.join(TICKERS), period='5y',
                               group_by='ticker', interval='1d')
            data.to_excel('cached_data.xlsx', index=True, header=True)

        for ticker in TICKERS:
            df = data[ticker].reset_index()
            close_history = df['Close']

            # Moving average for 10 and 50 days:
            ma10 = data[ticker]['Close'].rolling(window=10).mean()
            ma50 = data[ticker]['Close'].rolling(window=50).mean()

            # Exponential moving average for 10 and 50 days:
            ema10 = data[ticker]['Close'].ewm(span=10, adjust=False).mean()
            ema50 = data[ticker]['Close'].ewm(span=50, adjust=False).mean()

            # Normalize all data:
            close_history = close_history / close_history.abs().max()
            ma10 = ma10 / ma10.abs().max()
            ma50 = ma50 / ma50.abs().max()
            ema10 = ema10 / ema10.abs().max()
            ema50 = ema50 / ema50.abs().max()

            self.tickers[ticker] = {
                'df': df,
                'close_history': close_history,
                'ma10': ma10, 'ma50': ma50, 'ema10': ema10, 'ema50': ema50
            }

    def split_data(self) -> list:
        """ Split data by history and expected result. """

        data = []

        for ticker_data in self.tickers.values():
            data_len = len(ticker_data['close_history'])

            # take only shares which exist on the market for a long time already
            if data_len > 100:
                for i in range(0, data_len-self.long_period - 3, 3 + self.short_period):
                    data_points = ticker_data['close_history'][i:i + 3].values.tolist()
                    ma10 = ticker_data['ma10'][i:i + 3]
                    ma50 = ticker_data['ma50'][i:i + 3]
                    ema10 = ticker_data['ema10'][i:i + 3]
                    ema50 = ticker_data['ema50'][i:i + 3]

                    short_results = ticker_data['close_history'][i + 3:i + 3 + self.short_period]
                    long_results = ticker_data['close_history'][i + 3:i + 3 + self.long_period]

                    reward = 0

                    short_results_good = False
                    if data_points[-1] * self.short_benefit < sum(short_results) / len(short_results):
                        short_results_good = True
                        reward += 1

                    long_results_good = False
                    if data_points[-1] * self.long_benefit < sum(long_results) / len(long_results):
                        long_results_good = True
                        reward += 1

                    data.append({
                        'close': data_points,
                        'ma_short': ma10,
                        'ma_long': ma50,
                        'ema_short': ema10,
                        'ema_long': ema50,
                        'short_results': short_results_good,
                        'long_results': long_results_good,
                        'reward': reward
                    })

        return data


data_dealer = Data()
res = data_dealer.split_data()

print(res[-1])
