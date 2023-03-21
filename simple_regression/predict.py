import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from plotly.subplots import make_subplots

import pandas as pd
import pandas_ta  # for TA magic


PREDICT_INTERVAL = 10
regressor = LinearRegression()
# regressor = MLPRegressor(learning_rate_init=0.01, solver='adam')


def calculate_metrics(df):
    # Add extra info to the data:
    df.ta.ema(length=3, append=True, col_names=('EMA3',))
    df.ta.ema(length=7, append=True, col_names=('EMA7',))
    df.ta.ema(length=20, append=True, col_names=('EMA20',))
    df.ta.ema(length=35, append=True, col_names=('EMA35',))

    df.ta.ema(length=7 * 8, append=True, col_names=('EMA70',))
    df.ta.ema(length=14 * 8, append=True, col_names=('EMA140',))
    df.ta.ema(length=21 * 8, append=True, col_names=('EMA200',))

    df.ta.sma(length=50, append=True, col_names=('SMA50',))
    df.ta.sma(length=200, append=True, col_names=('SMA200',))

    df.ta.sma(length=50*8, append=True, col_names=('SMA500',))
    df.ta.sma(length=200*8, append=True, col_names=('SMA2000',))

    df.ta.rsi(append=True, col_names=('RSI',))
    df['RSI'] /= 100

    df.ta.rsi(length=8*14, append=True, col_names=('RSI_LONG',))
    df['RSI_LONG'] /= 100

    df.ta.atr(append=True, col_names=('ATR',))
    df['ATR'] /= 100

    df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)   # length=34,

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    # Extract the features (all the other variables)
    df = df[200*8:].copy()

    return df


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
TICKERS.remove('CEG')
TICKERS.remove('ELV')

for ticker in TICKERS[:1]:
    # Collect data for trainining
    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period='700d', interval='1h')

    # future_df = pd.DataFrame(index=range(len(df), len(df) + PREDICT_INTERVAL))
    # df = pd.concat([df, future_df])

    df['mean'] = (df['High'] + df['Low']) / 2
    df['target'] = df.ta.sma(close=df['mean'], length=PREDICT_INTERVAL, offset=-PREDICT_INTERVAL)
    df = df[:-PREDICT_INTERVAL].copy()  # remove latest data since we can't predict future values of expected data here

    df = calculate_metrics(df)

    graph = make_subplots(rows=1, cols=1)
    graph.update_layout(title='', xaxis_rangeslider_visible=False)
    graph.add_scatter(y=df['Close'], mode='lines', name='Close',
                      line={'color': 'green'})
    graph.add_scatter(y=df['target'], mode='lines', name='target',
                      line={'color': 'red'})
    graph.show()

    y = df['target']
    X = df.drop(columns=['target', 'mean'])   # df[['SMA50', 'SMA200']]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # regressor = MLPRegressor(learning_rate_init=0.0001, solver='adam')
    # for i in range(0, len(X_train)-WINDOW):
    #     regressor.fit(X_train[i:i+WINDOW], y_train[i:i+WINDOW])
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error: {mse:.2f}')


# Check how the model perform:

df2 = pd.DataFrame()
df2 = df.ta.ticker('AAPL', period='700d', interval='1h')
# X = calculate_metrics(df2)


import statsmodels.api as sm

# Fit the SARIMA model
df2.ta.sma(length=50, append=True, col_names=('SMA50',))
model = sm.tsa.ARIMA(df2['SMA50'], order=(2, 1, 2), seasonal_order=(0, 1, 1, 12))
results = model.fit()

# Make forecasts
forecasts = results.forecast(steps=10)

# Print the forecasts
print(forecasts)

"""
graph = make_subplots(rows=1, cols=1)
graph.update_layout(title='', xaxis_rangeslider_visible=False)

# graph.add_candlestick(open=X['Open'], high=X['High'], low=X['Low'], close=X['Close'])

graph.add_scatter(y=X['Close'], mode='lines', name='Close',
                  line={'color': 'green'})

graph.add_scatter(y=X['predicted'], mode='lines', name='predicted',
                  line={'color': 'magenta'})
graph.add_scatter(y=X['target'], mode='lines', name='target',
                  line={'color': 'orange'})

graph.add_scatter(y=X['SMA_PREDICT_INTERVAL'], mode='lines', name='SMA_PREDICT_INTERVAL',
                  line={'color': 'blue'})

graph.show()
"""