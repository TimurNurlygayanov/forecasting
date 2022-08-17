
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

from sklearn.metrics import mean_squared_error, mean_absolute_error


mse_data = {}
rmse_data = {}
mae_data = {}

ticker = 'AMD'

for data_period in ['3mo', '6mo', '1y', '2y', '5y']:
    data = yf.download(ticker, period=data_period, group_by='ticker', interval='1d')
    ticker_data_original = data['Close']

    for prediction_period in [5, 10, 20, 30]:  #  prediction_period = 14
        working_days = 5 * prediction_period // 7

        rmse_with_gaps = []

        for gap in [15, 10, 5, 0]:
            ticker_data = ticker_data_original[:-gap] if gap else ticker_data_original
            print(len(ticker_data))

            past_data = ticker_data[:-working_days].reset_index()
            last_data = ticker_data[-working_days:].reset_index()

            last_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
            model = Prophet(changepoint_prior_scale=0.5)
            model.add_country_holidays(country_name='US')

            df = past_data
            df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

            with suppress_stdout_stderr():
                model.fit(df)

            future = model.make_future_dataframe(periods=prediction_period+1)
            # Remove weekends from prediction, because we don't have data for weekends:
            future = future[future['ds'].dt.weekday <= 4]
            forecast = model.predict(future)

            graph = go.Figure()
            graph.add_scatter(x=df['ds'], y=df['y'], mode='lines',
                              name=f'{ticker} Closed price')
            graph.add_scatter(x=last_data['ds'], y=last_data['y'], mode='lines',
                              name=f'{ticker} Closed price')
            graph.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                              name='Forecast price')

            last_data_length = len(last_data)
            real_forecast = forecast[-last_data_length:]

            graph.add_scatter(x=real_forecast['ds'], y=real_forecast['yhat'], mode='lines',
                              name='Forecast price')

            # mse = mean_squared_error(real_forecast['yhat'], last_data['y'])  # MSE
            rmse = mean_squared_error(real_forecast['yhat'], last_data['y'], squared=False)           # RMSE
            # mae = mean_absolute_error(real_forecast['yhat'], last_data['y']) # MAE

            # Recalculate as a percents from the price:
            rmse = 100.0 * rmse / (sum(real_forecast['yhat']) / len(real_forecast['yhat']))

            rmse_with_gaps.append(rmse)

        print(f'data: {data_period} prediction: {prediction_period}')
        print('RMSE', max(rmse_with_gaps))
        print('- ' * 30)

        rmse_data[f'{data_period}_{prediction_period}'] = {'metric': max(rmse_with_gaps), 'graph': graph}

min_rmse = sorted(rmse_data.items(), key=lambda item: item[1]['metric'])
for res in min_rmse:
    print('MIN RMSE', res[0], res[1]['metric'])
    res[1]['graph'].show()
    break
