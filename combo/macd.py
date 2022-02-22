
import yfinance as yf
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go

import pandas_ta as ta

from datetime import datetime
from datetime import timedelta
from utils import suppress_stdout_stderr


prediction_period = 14
working_days = 5 * prediction_period // 7


ticker = 'MSFT'
data = yf.download(ticker, period='5y',
                   group_by='ticker', interval='1d')
ticker_data = data['Close']


# Split data to cut last known 30 days and make a prediction for these days
# to compare prediction and real data for the last period:
past_data = ticker_data[:-working_days].reset_index()
last_data = ticker_data[-working_days:].reset_index()

past_data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)

print(past_data)

graph = go.Figure()
graph.add_scatter(x=past_data['Date'], y=past_data['Close'],
                  name=f'{ticker} Closed price')
graph.add_scatter(x=past_data['Date'], y=past_data['MACD_12_26_9'], mode='lines', name='MACD')
graph.add_scatter(x=past_data['Date'], y=past_data['MACDs_12_26_9'], mode='lines', name='signal')

colors = np.where(past_data['MACDh_12_26_9'] < 0, '#000', '#ff9900')
graph.add_bar(x=past_data['Date'], y=past_data['MACDh_12_26_9'], marker_color=colors, name='histogram')

graph.update_layout(height=1000, width=1500)

graph.update_xaxes(
    rangebreaks=[
        dict(bounds=['sat', 'mon'])
    ]
)
# graph.show()



def plot_macd(prices, macd, signal, hist):
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(prices)
    ax2.plot(macd, color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(signal, color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax2.bar(prices.index[i], hist[i], color = '#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], color = '#26a69a')

    plt.legend(loc = 'lower right')
    plt.show()


plot_macd(past_data['Close'], past_data['MACD_12_26_9'], past_data['MACDs_12_26_9'], past_data['MACDh_12_26_9'])
