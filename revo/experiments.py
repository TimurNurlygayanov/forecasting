
# https://trading-data-analysis.pro/trading-algorithm-that-doesnt-work-37e747f4c6a6
# https://towardsdatascience.com/making-a-stock-screener-with-python-4f591b198261
#

import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
from sktime.forecasting.model_selection import temporal_train_test_split
# import datetime
# import pyaf.ForecastEngine as autof

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS

import catboost as cb


if __name__ == '__main__':
    graph = go.Figure()

    # ticker = 'CSCO'
    ticker = 'AAPL'
    # ticker = 'V'

    data = yf.download(ticker, period='2y',
                       group_by='ticker', interval='1d')

    data = data.reset_index()
    data['Close'] = data['Close'] - data['Close'].min() * 0.9
    data['Close'] = data['Close'] / data['Close'].max()
    # data['Close'] = data['Close'].rolling(window=2).mean()

    # data['Volume'] = data['Volume'] / data['Volume'].max()
    # data['Close'] = np.log(data['Close'])

    data, data_after_prediction = temporal_train_test_split(data, test_size=50)
    data['Date'] = data['Date'].dt.tz_localize(None)
    data_after_prediction['Date'] = data_after_prediction['Date'].dt.tz_localize(None)

    data['MA_short'] = data['Close'].rolling(window=21).mean()
    data['MA_short'] = data['MA_short'].fillna(0)
    data['MA_long'] = data['Close'].rolling(window=50).mean()
    data['MA_long'] = data['MA_long'].fillna(0)

    # Future data:
    graph.add_scatter(y=data_after_prediction['Close'], x=data_after_prediction['Close'].index,
                      mode='lines', name='Future', line={'color': 'brown'})

    graph.add_scatter(y=data['Close'], mode='lines', name='Price', line={'color': 'green'})

    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    data['RSI'] = data['RSI'] / 100.0

    """
    bbands = ta.bbands(data['Close'], length=20, std=2.3)
    data['L'] = bbands['BBL_20_2.3']
    data['M'] = bbands['BBM_20_2.3']
    data['U'] = bbands['BBU_20_2.3']
    """

    data['EMA_short'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=50, adjust=False).mean()

    # macd = ta.macd(data['Close'], fast=12, slow=26, signal=9, append=True)

    # graph.add_scatter(y=macd['MACD_12_26_9'], mode='lines', name='MACD')
    # graph.add_scatter(y=macd['MACDs_12_26_9'], mode='lines', name='MACD_signal')
    # macd['MACDh_12_26_9'] = macd['MACDh_12_26_9'] / (2*macd['MACDh_12_26_9'].max()) + 0.5
    # graph.add_scatter(y=macd['MACDh_12_26_9'], mode='lines', name='MACD_histogram')

    graph.add_scatter(y=data['MA_short'], mode='lines', name='MA_short')
    graph.add_scatter(y=data['MA_long'], mode='lines', name='MA_long')
    # graph.add_scatter(y=data['EMA_short'], mode='lines', name='EMA_short')
    # graph.add_scatter(y=data['EMA_long'], mode='lines', name='EMA_long')

    graph.add_scatter(y=data['RSI'], mode='lines', name='RSI')

    # data['RSI_MA_short'] = data['RSI'].rolling(window=20).mean()
    # graph.add_scatter(y=data['RSI_MA_short'], mode='lines', name='RSI MA')

    # graph.add_scatter(y=data['L'], mode='lines', name='L')
    # graph.add_scatter(y=data['M'], mode='lines', name='M')
    # graph.add_scatter(y=data['U'], mode='lines', name='U')

    # graph.add_scatter(y=data['MA_short'], mode='lines', name='MA_short')
    # graph.add_scatter(y=data['MA_long'], mode='lines', name='MA_long')
    # graph.add_scatter(y=data['Volume'], mode='lines', name='Volume')

    """
    z = np.polyfit(data['Close'].index, data['Close'], 1)
    p = np.poly1d(z)
    
    trend_period = len(data['Close']) // 5
    for i in range(0, len(data['Close']), trend_period):
        z = np.polyfit(data['Close'][i:i+trend_period].index, data['Close'][i:i+trend_period], 1)
        p = np.poly1d(z)
        graph.add_scatter(y=p(data['Close'][i:i+trend_period].index), x=data['Close'][i:i+trend_period].index,
                          mode='lines', name='Trend', line={'color': 'grey'})
    
    """

    graph.show()

