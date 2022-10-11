import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.croston import Croston
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.arima import AutoARIMA


def get_prediction(ticker_data):
    y_train, y_test = temporal_train_test_split(ticker_data['Close'], test_size=30)

    fh = ForecastingHorizon(y_test.index, is_relative=False)

    # forecaster = ThetaForecaster(sp=5)  # monthly seasonal periodicity
    # forecaster = NaiveForecaster(strategy="last", sp=5)
    # forecaster = SARIMAX(order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 3))
    forecaster = ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))

    forecaster.fit(y_train, fh=[1, 2, 3])

    y_pred = forecaster.predict(fh)

    return y_test, y_pred


graph = go.Figure()

ticker = 'CSCO'
# ticker = 'MSFT'

data = yf.download(ticker, period='1y',
                   group_by='ticker', interval='1d')

data = data.reset_index()
data['Close'] = data['Close'] - data['Close'].min() * 0.9
data['Close'] = data['Close'] / data['Close'].max()
# data['Close'] = np.log(data['Close'])

graph.add_scatter(y=data['Close'], mode='lines', name='Price')

data['MA_short'] = data['Close'].rolling(window=5).mean()
data['MA_short'] = data['MA_short'].fillna(0)
graph.add_scatter(y=data['MA_short'], mode='lines', name='MA5')

test_data, forecast = get_prediction(data)
graph.add_scatter(y=test_data, x=test_data.index, mode='lines', name='Real Data')
graph.add_scatter(y=forecast, x=forecast.index, mode='lines', name='Forecast')


graph.show()

