
import numpy as np
import pandas as pd
from os import path
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go

import ta

from datetime import datetime
from datetime import timedelta
from utils import suppress_stdout_stderr


prediction_period = 20
working_days = 5 * prediction_period // 7


data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])

TICKERS = set([t[0] for t in data.columns.values])

for ticker in TICKERS:
    ticker_data = data[ticker]['Close']

    if len(ticker_data) < 365:
        print(f'{ticker} has not enough data to make predictions')
        continue

    if np.isnan(ticker_data.values[-1]):
        print(f'{ticker} has not data (NaN) to make predictions')
        continue

    # Split data to cut last known 30 days and make a prediction for these days
    # to compare prediction and real data for the last period:
    past_data = ticker_data[:-working_days].reset_index()
    last_data = ticker_data[-working_days:].reset_index()
    last_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

    ticker_data_bb = ta.utils.dropna(past_data)
    indicator_bb = ta.volatility.BollingerBands(close=ticker_data_bb["Close"])

    # Add Bollinger Bands features
    bb_bbm = indicator_bb.bollinger_mavg()
    bb_bbh = indicator_bb.bollinger_hband()
    bb_bbl = indicator_bb.bollinger_lband()

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

    forecast_to_grow = forecast['yhat'].values[-10] < forecast['yhat'].values[-1]
    trend_up = forecast['trend'].values[-10] < forecast['trend'].values[-1]

    # Draw a graph to show the forecast from the last month
    # for ~30 days and real data for the last 30 days
    graph = go.Figure()
    graph.add_scatter(x=df['ds'], y=df['y'],
                      name=f'{ticker} Closed price')
    graph.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                      name='Forecast price')
    graph.add_scatter(x=last_data['ds'], y=last_data['y'], mode='lines',
                      name=f'{ticker} Closed price future fact')
    # graph.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
    #                   name='Minimium forecasted price')
    graph.add_scatter(x=forecast['ds'], y=forecast['trend'], mode='lines',
                      name='Trend')
    graph.add_scatter(x=df['ds'], y=bb_bbh, name='Bolinger High')
    graph.add_scatter(x=df['ds'], y=bb_bbl, name='Bolinger Low')

    graph.update_layout(height=1000, width=1500)
    graph.show()
