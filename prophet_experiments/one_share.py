
import yfinance as yf
import numpy as np
import pandas as pd
from os import path
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from datetime import timedelta
from utils import suppress_stdout_stderr


prediction_period = 14
working_days = 5 * prediction_period // 7


ticker = 'FHN'
data = yf.download(ticker, period='2y',
                   group_by='ticker', interval='1d')
ticker_data = data['Close']


# Split data to cut last known 30 days and make a prediction for these days
# to compare prediction and real data for the last period:
past_data = ticker_data[:-working_days].reset_index()
last_data = ticker_data[-working_days:].reset_index()

last_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
model = Prophet(changepoint_prior_scale=0.5)
model.add_country_holidays(country_name='US')

df = past_data
df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

with suppress_stdout_stderr():
    model.fit(df)

future = model.make_future_dataframe(periods=prediction_period+3)
# Remove weekends from prediction, because we don't have data for weekends:
future = future[future['ds'].dt.weekday <= 4]
forecast = model.predict(future)

ema10 = ticker_data.ewm(span=20, adjust=False).mean()
ema30 = ticker_data.ewm(span=50, adjust=False).mean()
ema200 = ticker_data.ewm(span=100, adjust=False).mean()

ma14 = ticker_data.rolling(window=14).mean()

graph = go.Figure()
graph.add_scatter(x=df['ds'], y=df['y'],
                  name=f'{ticker} Closed price')
"""
graph.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                  name='Forecast price')
graph.add_scatter(x=last_data['ds'], y=last_data['y'], mode='lines',
                  name=f'{ticker} Closed price future fact')
graph.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                  name='Minimium forecasted price')
graph.add_scatter(x=forecast['ds'], y=forecast['trend'], mode='lines',
                  name='Trend')
"""
graph.add_scatter(x=df['ds'], y=ema10, mode='lines', name='EMA 10')
graph.add_scatter(x=df['ds'], y=ema30, mode='lines', name='EMA 30')
graph.add_scatter(x=df['ds'], y=ema200, mode='lines', name='EMA 200')

graph.add_scatter(x=df['ds'], y=ma14, mode='lines', name='MA 14')

graph.update_layout(height=1000, width=1500)
graph.show()
