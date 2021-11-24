#!/usr/bin/python3
# -*- encoding=utf8 -*-

from os import path
import yfinance as yf
import pandas as pd
import numpy as np


with open('data/revolut_tickers.txt', 'r', encoding='utf-8') as tickers_file:
    TICKERS = tickers_file.readlines()

TICKERS = [t.strip() for t in TICKERS if t]
TICKERS = TICKERS[:10]

data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])
else:
    data = yf.download(TICKERS, period='5y',
                       group_by='ticker', interval='1d')
    data.to_excel(data_file_name, index=True, header=True)

for ticker in TICKERS:
    ticker_data = data[ticker]
    # Log2 from real price to make it smooth
    ticker_data['CloseLog'] = np.log(ticker_data['Close'])

    ticker_data.to_excel(f'data/tickers/{ticker}.xlsx', index=True, header=True)
