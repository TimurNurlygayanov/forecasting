
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from datetime import timedelta

import ta


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

    ema_short = past_data['Close'].ewm(span=10, adjust=False).mean()
    ema_long = past_data['Close'].ewm(span=200, adjust=False).mean()

    ma3 = ticker_data.rolling(window=3).mean()
    ma21 = ticker_data.rolling(window=14).mean()

    signal = False
    """
    if past_data['Close'].values[-1] > ema_long.values[-1]:
        if ema_short.values[-1] > ema_long.values[-1]:       # ema 10 higher ema200
                for i in range(2, 6):
                    if past_data['Close'].values[-i] < ema_long.values[-i]:
                        signal = True
                        break
    """

    ticker_data_bb = ta.utils.dropna(past_data)
    indicator_bb = ta.volatility.BollingerBands(close=ticker_data_bb["Close"])

    # Add Bollinger Bands features
    bb_bbm = indicator_bb.bollinger_mavg()
    bb_bbh = indicator_bb.bollinger_hband()
    bb_bbl = indicator_bb.bollinger_lband()

    """
    if past_data['Close'].values[-1] > bb_bbl.values[-1]:
        if past_data['Close'].values[-1] < ema_long.values[-1]:
            for i in range(2, 7):
                if past_data['Close'].values[-i] < bb_bbl.values[-1]:
                    signal = True
    """

    # ignore share with high price
    if past_data['Close'].values[-1] > 30:
        continue

    if past_data['Close'].values[-1] > ema_long.values[-1]:
        for i in range(2, 3):
            if past_data['Close'].values[-i] < ema_long.values[-i]:
                signal = True
                break

    if signal:

        graph = go.Figure()
        graph.add_scatter(x=past_data['Date'], y=past_data['Close'],
                          name=f'{ticker} Closed price')
        graph.add_scatter(x=last_data['Date'], y=last_data['Close'], mode='lines',
                          name=f'{ticker} Closed price future fact')

        graph.add_scatter(x=past_data['Date'], y=ema_long, mode='lines', name='EMA 150')
        graph.add_scatter(x=past_data['Date'], y=ema_short, mode='lines', name='EMA 10')

        # graph.add_scatter(x=past_data['Date'], y=ma3, mode='lines', name='MA 3')
        # graph.add_scatter(x=past_data['Date'], y=ma21, mode='lines', name='MA 21')

        # graph.add_scatter(x=past_data['Date'], y=bb_bbh, name='Bolinger High')
        # graph.add_scatter(x=past_data['Date'], y=bb_bbl, name='Bolinger Low')

        graph.update_layout(height=1000, width=1500)
        graph.show()

        print('* ' * 20)
        print(f'Buy: {ticker}')

        found_shares += 1
        if found_shares >= 10:
            exit(1)
