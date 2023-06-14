from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from os import path
import pandas as pd
import pandas_ta  # for TA magic
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

bet_period = 10
sample_period = 200


def parse_data(ticker='CSCO'):
    file_name = f'rl/data/daily_{ticker}.xlsx'

    if path.exists(file_name):
        # print('Reading data from cache...')

        df = pd.read_excel(file_name, index_col=0)
    else:
        # print('Getting data... ')

        df = pd.DataFrame()
        df = df.ta.ticker(ticker, period='700d', interval='1h')
        df = df.reset_index()

        # print('Calculating data... ')

        df.ta.ema(length=3, append=True, col_names=('EMA3',))
        df.ta.ema(length=7, append=True, col_names=('EMA7',))
        df.ta.ema(length=20, append=True, col_names=('EMA20',))

        df.ta.wma(length=9, append=True, col_names=('WMA9',))
        df.ta.wma(length=14, append=True, col_names=('WMA14',))

        df.ta.cti(append=True, col_names=('CTI',))

        df.ta.sma(length=50, append=True, col_names=('SMA50',))
        df.ta.sma(length=200, append=True, col_names=('SMA200',))

        df.ta.rsi(append=True, col_names=('RSI',))
        df.ta.atr(append=True, col_names=('ATR',))
        df.ta.supertrend(append=True, length=34, multiplier=3.0,
                         col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
        df.ta.supertrend(append=True, length=10, multiplier=2.0,
                         col_names=('S_trend2', 'S_trend_d2', 'S_trend_l2', 'S_trend_s2',))
        df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)
        df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

        df['t1'] = np.where(df['EMA3'] > df['EMA7'], 1, 0)
        df['t2'] = np.where(df['EMA7'] > df['EMA20'], 1, 0)
        df['t3'] = np.where(df['RSI'] < 35, 1, 0)

        for x in ['Close', 'Open', 'High', 'Low', 'S_trend', 'S_trend2', 'EMA3', 'EMA7', 'EMA20', 'SMA50', 'L', 'U', 'WMA9', 'WMA14', 'ATR']:
            df[x] /= df['SMA200']

        df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        df.to_excel(file_name, index=True, header=True)

    df = df.drop('Volume', axis=1)
    df = df.drop('Datetime', axis=1)
    df = df.drop('Dividends', axis=1)
    df = df.drop('Stock Splits', axis=1)
    df = df.drop('S_trend_l', axis=1)
    df = df.drop('S_trend_s', axis=1)
    df = df.drop('S_trend_l2', axis=1)
    df = df.drop('S_trend_s2', axis=1)

    df = df[200:].copy()

    return df


def get_data(df):
    y1 = []
    y2 = []

    df['target'] = 0

    for index, row in df[:-sample_period].iterrows():
        current_price = df['Close'][index]

        low_stop_loss = current_price
        high_close = 0

        for i in range(1, bet_period):
            if df['Low'][index + i] < low_stop_loss:
                low_stop_loss = df['Low'][index + i]

            if df['Close'][index + i] > high_close:
                high_close = df['Close'][index + i]

        y1.append(low_stop_loss)
        y2.append(high_close)

    df = df[:-sample_period].copy()

    return df, y1, y2


# Load or create your dataset
df = parse_data('MSFT')
to_predict = df[-sample_period:].copy()
X, y1, y2 = get_data(df)  # Replace with your own dataset loading code

# Split the data into training and testing sets
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.1, random_state=42)
# Create and train the CatBoost regression model
model = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=10)
model.fit(X_train, y2_train, verbose=False)
# Make predictions on the test set
y2_pred = model.predict(to_predict)

# Split the data into training and testing sets
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.1, random_state=42)
# Create and train the CatBoost regression model
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=3)
model.fit(X_train, y1_train, verbose=False)
# Make predictions on the test set
y1_pred = model.predict(to_predict)

to_predict['y1'] = y1_pred
to_predict['y2'] = y2_pred


def draw(df):
    graph = make_subplots(rows=1, cols=1, shared_xaxes=True)
    graph.update_layout(title='D', xaxis_rangeslider_visible=False)

    graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

    graph.add_scatter(y=df['y1'], mode='lines', name='Stop Loss')
    graph.add_scatter(y=df['y2'], mode='lines', name='Take Profit')

    graph.show()


draw(to_predict)
