#
# Ищем такие акции, у которых MA3 пересекает MA21 снизу вверх,
# то есть сейчас MA3 выше MA21, но несколько дней назад (3 дня) была ниже
# При этом линия тренда цены по Prophnet должна быть выше текущей цены.
#

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


def check_mooving_average_combo(ma_short: list, ma_long: list):
    # Ищем такие акции, у которых MA3 пересекает MA21 снизу вверх,
    # то есть сейчас MA3 выше MA21, но несколько дней назад (3 дня) была ниже
    if ma_long[-1] < ma_short[-1]:
        if ma_long[-2] > ma_short[-2] or ma_long[-3] > ma_short[-3] or ma_long[-4] > ma_short[-4]:
            return True

    return False


def check_ema_and_price(ema10, price):

    for i in range(1, 10):
        if ema10[-i] > price[-i]:
            return False

    return True

prediction_period = 20
working_days = 5 * prediction_period // 7


data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])

TICKERS = set([t[0] for t in data.columns.values])

for ticker in TICKERS:
    ticker_data = data[ticker]['Close'][:-working_days]

    print(ticker)

    if len(ticker_data) < 100:
        print(f'{ticker} has not enough data to make predictions')
        continue

    if np.isnan(ticker_data.values[-1]):
        print(f'{ticker} has not data (NaN) to make predictions')
        continue

    ma3 = ticker_data.rolling(window=3).mean()
    ma21 = ticker_data.rolling(window=14).mean()

    ema10 = ticker_data.ewm(span=10, adjust=False).mean()
    ema30 = ticker_data.ewm(span=30, adjust=False).mean()

    model = Prophet(changepoint_prior_scale=0.5)
    model.add_country_holidays(country_name='US')

    df = ticker_data.reset_index()
    df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

    with suppress_stdout_stderr():
        model.fit(df)

    future = model.make_future_dataframe(periods=prediction_period+3)
    # Remove weekends from prediction, because we don't have data for weekends:
    future = future[future['ds'].dt.weekday <= 4]
    forecast = model.predict(future)

    trend_up = forecast['trend'].values[-20] < forecast['trend'].values[-1]
    ma_signal_to_buy = check_mooving_average_combo(ma3.values, ma21.values)
    # ema_and_price = check_ema_and_price(ema10.values, df['y'].values)
    # ema_signal_to_buy = check_mooving_average_combo(ema10.values, ema30.values)
    price_lower_trend = df['y'].values[-1] < forecast['trend'].values[-1]

    # if trend_up and ma_signal_to_buy and price_lower_trend:
    if True:
        last_data = data[ticker]['Close'][-working_days:].reset_index()
        last_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

        # if trend_up and ema_signal_to_buy:
        # Draw a graph to show the forecast from the last month
        # for ~30 days and real data for the last 30 days
        graph = go.Figure()
        graph.add_scatter(x=df['ds'], y=df['y'],
                          name=f'{ticker} Closed price')
        graph.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                          name='Forecast price')
        graph.add_scatter(x=last_data['ds'], y=last_data['y'], mode='lines',
                          name=f'{ticker} Closed price future fact')
        graph.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                          name='Minimium forecasted price')
        graph.add_scatter(x=forecast['ds'], y=forecast['trend'], mode='lines',
                          name='Trend')
        graph.add_scatter(x=df['ds'], y=ma3, name='MA 3', mode='lines')
        graph.add_scatter(x=df['ds'], y=ma21, name='MA 21', mode='lines')
        graph.add_scatter(x=df['ds'], y=ema10, name='EMA 10', mode='lines')
        graph.add_scatter(x=df['ds'], y=ema30, name='EMA 30', mode='lines')

        graph.update_layout(height=1000, width=1500)
        graph.show()

        exit(1)
