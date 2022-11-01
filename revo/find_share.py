
from pathlib import Path

import numpy as np
import pandas_ta as ta
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import TrendForecaster
from scipy.signal import argrelextrema


def get_trend(data_sample):
    z = np.polyfit(data_sample.index, data_sample, 1)
    p = np.poly1d(z)
    trend = p(data_sample.index)

    return trend


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t]



data_file = 'data.csv'
if Path(data_file).is_file():
    all_tickers_data = pd.read_csv(data_file, index_col=0, header=[0, 1])
else:
    all_tickers_data = yf.download(TICKERS, period='2y',
                                   group_by='ticker', interval='1d')
    all_tickers_data.to_csv(data_file)

t = 0
for ticker in TICKERS:
    # print(f'Checking {ticker}')

    full_data = all_tickers_data[ticker]

    if np.isnan(full_data['Close'].values[-2]):
        continue

    full_data = full_data.reset_index()
    full_data['Close'] = full_data['Close'] - full_data['Close'].min() * 0.9
    full_data['Close'] = full_data['Close'] / full_data['Close'].max()
    # full_data['Close'] = full_data['Close'].rolling(window=2).mean()  # MA2 for close price to make it more smooth

    # full_data['Volume'] = full_data['Volume'] - full_data['Volume'].min() * 0.9
    # full_data['Volume'] = full_data['Volume'] / full_data['Volume'].max()

    data, data_after_prediction = temporal_train_test_split(full_data, test_size=50)

    print(ticker, len(data))

    data['MA_short'] = data['Close'].rolling(window=12).mean()
    data['MA_short'] = data['MA_short'].fillna(0)

    data['MA_long'] = data['Close'].rolling(window=50).mean()
    data['MA_long'] = data['MA_long'].fillna(0)

    # data['EMA'] = ta.hlc3(high=data['High'], low=data['Low'], close=data['Close'], length=21)

    data['MA_long100'] = data['Close'].rolling(window=100).mean()

    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    data['RSI'] = data['RSI'].fillna(0) / 100.0
    data['RSI_ma'] = data['RSI'].rolling(window=7).mean()

    bbands = ta.bbands(data['Close'], length=34)
    data['L'] = bbands['BBL_34_2.0']
    data['M'] = bbands['BBM_34_2.0']
    data['U'] = bbands['BBU_34_2.0']

    # Find local minimums on RSI:
    ilocs_min = argrelextrema(data['RSI'].values, np.less_equal, order=10)[0]
    ilocs_max = argrelextrema(data['RSI'].values, np.greater_equal, order=10)[0]

    last_rsi_max = data['RSI'].values[ilocs_max[-1]]
    min_rsi = data['RSI'].values[ilocs_min[-1]]

    rsi_signal = False
    if 3 < len(data['RSI']) - ilocs_min[-1] < 20:   # and last_rsi_max > 0.6:
        rsi_signal = True

    ma_signal = False
    if data['Close'].values[-1] < data['MA_long'].values[-1]:
        if abs(data['MA_long'].values[-1] - data['MA_long100'].values[-1]) > 0.05:
            ma_signal = True

    bb_signal = False
    if data['L'].values[-1] > data['Close'].values[-1]:
        bb_signal = True

    rvi = ta.rvi(data['Close'], high=data['High'], low=data['Low'])
    rvi = rvi / 100.0

    if bb_signal:
        graph = go.Figure()
        graph.update_layout(title=ticker)

        graph.add_scatter(x=ilocs_min, y=data['RSI'].values[ilocs_min], mode='markers')

        # Old data:
        graph.add_scatter(y=data['Close'], mode='lines', name='Price',
                          line={'color': 'green', 'width': 3})

        # graph.add_scatter(y=data['RSI'], mode='lines', name='RSI',
        #                   line={'color': 'black'})
        # graph.add_scatter(y=data['RSI_ma'], mode='lines', name='RSI_ma',
        #                   line={'color': 'red'})

        # Future data:
        graph.add_scatter(y=data_after_prediction['Close'], x=data_after_prediction['Close'].index,
                          mode='lines', name='Future', line={'color': 'brown'})

        # Moving averages:
        # graph.add_scatter(y=data['MA_short'], mode='lines', name='MA_short',
        #                   line={'color': 'orange', 'width': 2})
        graph.add_scatter(y=data['MA_long'], mode='lines', name='MA_long',
                          line={'color': 'magenta', 'width': 3})

        # graph.add_scatter(y=data['EMA'], mode='lines', name='EMA',
        #                   line={'color': 'grey'})

        # graph.add_scatter(y=data['MA_long100'], mode='lines', name='MA_long100',
        #                   line={'color': 'blue', 'width': 3})

        # graph.add_scatter(y=rvi, mode='lines', name='RVI_14',
        #                   line={'color': 'blue'})

        """
        graph.add_scatter(y=data['MA_long'], mode='lines', name='MA_long',
                          line={'color': 'rgb(20, 20, 219)'})
        graph.add_scatter(y=data['MA_long100'], mode='lines', name='MA_long100',
                          line={'color': 'orange'})
        """

        graph.add_scatter(y=data['L'], mode='lines', name='L')
        graph.add_scatter(y=data['U'], mode='lines', name='U')

        graph.show()

        t += 1

        if t > 5:
            exit(1)
