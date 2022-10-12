
import yfinance as yf
import plotly.graph_objects as go
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import TrendForecaster


with open('revolut_tickers.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS]
# TICKERS = TICKERS[:100]


all_tickers_data = yf.download(TICKERS, period='1y',
                               group_by='ticker', interval='1d')

for ticker in TICKERS:
    print(f'Checking {ticker}')

    full_data = all_tickers_data[ticker]

    full_data = full_data.reset_index()
    full_data['Close'] = full_data['Close'] - full_data['Close'].min() * 0.9
    full_data['Close'] = full_data['Close'] / full_data['Close'].max()

    data, data_after_prediction = temporal_train_test_split(full_data)

    data['MA_short'] = data['Close'].rolling(window=3).mean()
    data['MA_short'] = data['MA_short'].fillna(0)

    data['MA_long'] = data['MA_short'].rolling(window=21).mean()
    data['MA_long'] = data['MA_long'].fillna(0)

    if data['Close'].values[-1] > data['MA_long'].values[-1]:
        if data['MA_long'].values[-1] > data['MA_short'].values[-1]:
            forecaster = TrendForecaster()
            forecaster.fit(data['MA_short'])
            predicted_trend = forecaster.predict(fh=[1, 2, 3])

            # Check shares only if we predict positive trend for price
            if predicted_trend.values[1] > predicted_trend.values[-1]:
                print(predicted_trend)
                continue

            for i in range(2, 5):
                if data['Close'].values[-i] < data['MA_long'].values[-i]:
                    graph = go.Figure()
                    graph.update_layout(title=ticker)

                    # Old data:
                    graph.add_scatter(y=data['Close'], mode='lines', name='Price',
                                      line={'color': 'rgb(19, 161, 50)'})
                    # Future data:
                    graph.add_scatter(y=data_after_prediction['Close'], x=data_after_prediction['Close'].index,
                                      mode='lines', name='Future', line={'color': 'rgb(219, 20, 219)'})

                    # Moving averages:
                    graph.add_scatter(y=data['MA_short'], mode='lines', name='MA_short',
                                      line={'color': 'rgb(245, 59, 59)'})
                    graph.add_scatter(y=data['MA_long'], mode='lines', name='MA_long',
                                      line={'color': 'rgb(20, 20, 219)'})

                    graph.add_scatter(y=predicted_trend, x=predicted_trend.index, mode='lines', name='Trend',
                                      line={'color': 'black'})

                    graph.show()

                    break
