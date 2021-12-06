#!/usr/bin/python3
# -*- encoding=utf8 -*-

import random
from os import path
import yfinance as yf
import pandas as pd
import numpy as np


with open('data/revolut_tickers.txt', 'r', encoding='utf-8') as tickers_file:
    TICKERS = tickers_file.readlines()

TICKERS = [t.strip() for t in TICKERS if t]
random.shuffle(TICKERS)
TICKERS = TICKERS[:100]
# TICKERS = ['TSLA', 'MSFT', 'IBM', 'NVDA', 'FB', 'GOOGL', 'SEMR']
TICKERS = ['BJ', 'BLNK', 'BNTX', 'CARS', 'COCO', 'DLTR', 'GME', 'GOEV', 'HYZN', 'LCID', 'M', 'MRNA', 'NBEV', 'NVAX', 'RBLX', 'TAL', 'VG']

data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])
else:
    data = yf.download(TICKERS, period='5y',
                       group_by='ticker', interval='1d')
    data.to_excel(data_file_name, index=True, header=True)
