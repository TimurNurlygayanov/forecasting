# !/usr/bin/python3
# -*- encoding=utf8 -*-

import datetime
import pandas as pd

from os import path
import yfinance as yf


START_DATE = datetime.datetime(2021, 9, 1)
END_DATE = datetime.datetime.now().date().isoformat()  # today
START_DATE = str(START_DATE).split(' ')[0]
END_DATE = str(END_DATE).split(' ')[0]

with open('data/revolut_tickers.txt', 'r') as f:
    TICKERS = f.readlines()
TICKERS = [t.strip() for t in TICKERS if t]

data = yf.download(TICKERS,
                   start=START_DATE, end=END_DATE,
                   group_by='ticker', interval='1d')

for ticker in TICKERS:
    close_price = data[ticker]['Close'].values.tolist()

    min_price = close_price[-20]
    max_price = max(close_price[-19:])
    profit = (max_price - min_price) / min_price
    if profit > 0.15:
        print('TOP gainer {0} {1:.2f}%  {2}-{3}'.format(ticker, profit,
                                                        min_price, max_price))
