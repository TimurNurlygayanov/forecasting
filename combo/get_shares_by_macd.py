
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from datetime import timedelta

import pandas_ta as ta


prediction_period = 60
working_days = 5 * prediction_period // 7
found_shares = 0


data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])

TICKERS = set([t[0] for t in data.columns.values])

for ticker in TICKERS:
    print(f'Checking {ticker}...')

    ticker_data = data[ticker]['Close']

    if len(ticker_data) < 100:
        print(f'{ticker} has not enough data to make predictions')
        continue

    if np.isnan(ticker_data.values[-1]):
        print(f'{ticker} has not data (NaN) to make predictions')
        continue

    # Split data to cut last known 30 days and make a prediction for these days
    # to compare prediction and real data for the last period:
    past_data = ticker_data[:-working_days].reset_index()
    last_data = ticker_data[-working_days:].reset_index()

    ema_long = past_data['Close'].ewm(span=200, adjust=False).mean()

    past_data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)

    # ignore share with high price
    if past_data['Close'].values[-1] > 30:
        continue

    # ignore shares if price under long EMA
    signal_ema = True
    # if past_data['Close'].values[-1] < ema_long.values[-1]
    for i in range(20):
        if past_data['Close'].values[-i] < ema_long.values[-i]:
            signal_ema = False

    signal = False
    if past_data['MACDs_12_26_9'].values[-1] < past_data['MACD_12_26_9'].values[-1]:
        for i in range(2, 6):
            if past_data['MACDs_12_26_9'].values[-i] > past_data['MACD_12_26_9'].values[-i]:
                signal = True

    """
    signal_macd = False
    for i in range(5):
        if past_data['MACD_12_26_9'].values[-i] < -0.5:
            signal_macd = True
    """
    if signal and signal_ema:

        graph = go.Figure()
        graph.add_scatter(x=past_data['Date'], y=past_data['Close'],
                          name=f'{ticker} Closed price')
        graph.add_scatter(x=last_data['Date'], y=last_data['Close'], mode='lines',
                          name=f'{ticker} Closed price future fact')

        graph.add_scatter(x=past_data['Date'], y=ema_long, mode='lines', name='EMA 200')

        graph.add_scatter(x=past_data['Date'], y=past_data['MACD_12_26_9'], mode='lines', name='MACD')
        graph.add_scatter(x=past_data['Date'], y=past_data['MACDs_12_26_9'], mode='lines', name='signal')

        graph.update_layout(height=1000, width=1500)
        graph.show()

        print('* ' * 20)
        print(f'Buy: {ticker}')

        found_shares += 1
        if found_shares >= 10:
            exit(1)
