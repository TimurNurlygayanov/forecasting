import pandas as pd
import pandas_ta  # for TA magic
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots

ticker = 'MSFT'
prediction_window = 15
sample_window = 200

df = pd.DataFrame()
df = df.ta.ticker(ticker, period='700d', interval='1h')
df = df.reset_index()

# Add columns for linear regression predictions
df['Max_Close_Prediction'] = pd.Series(dtype=float)
df['Min_Low_Prediction'] = pd.Series(dtype=float)
df['y1'] = pd.Series(dtype=float)
df['y2'] = pd.Series(dtype=float)

# Define the features and target variables
features = ['Close', 'Open', 'High', 'Low', 'Volume']  # Add additional features as needed
target_max_close = df['Close'].rolling(window=prediction_window).max().shift(-prediction_window).values
target_min_low = df['Low'].rolling(window=3).min().shift(-prediction_window).values

# Perform linear regression for Max Close Price
X_max_close = df[features].values[:-prediction_window]
y_max_close = target_max_close[:-prediction_window]

lr_max_close = LinearRegression()
lr_max_close.fit(X_max_close, y_max_close)
df['Max_Close_Prediction'][:-prediction_window] = lr_max_close.predict(X_max_close)

# Perform linear regression for Min Low Price
X_min_low = df[features].values[:-prediction_window]
y_min_low = target_min_low[:-prediction_window]

lr_min_low = LinearRegression()
lr_min_low.fit(X_min_low, y_min_low)
df['Min_Low_Prediction'][:-prediction_window] = lr_min_low.predict(X_min_low)

# Make predictions for the latest data
X_latest = df[features].values[-sample_window:]
df['y2'][-sample_window:] = lr_max_close.predict(X_latest)
df['y1'][-sample_window:] = lr_min_low.predict(X_latest)


def draw(df):
    graph = make_subplots(rows=1, cols=1, shared_xaxes=True)
    graph.update_layout(title='D', xaxis_rangeslider_visible=False)

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

    graph.add_scatter(y=df['y1'], mode='lines', name='Stop Loss')
    graph.add_scatter(y=df['y2'], mode='lines', name='Take Profit')

    graph.show()


draw(df[-sample_window:])
