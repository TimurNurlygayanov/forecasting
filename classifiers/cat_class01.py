# After the model was trained on specific ticker, it shows very good results for this ticker
# but if it was trained on a different ticker, it doesn't show good results.

from sklearn.linear_model import LogisticRegression
import numpy as np
from os import path
from catboost import CatBoostClassifier
import pandas as pd
import pandas_ta  # for TA magic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings


# Set the warnings filter to "ignore"
warnings.filterwarnings("ignore")

"""
stop_loss = 0.95
take_profit = 1.1
bet_period = 30 * 8
"""

stop_loss = 0.99
take_profit = 1.02
bet_period = 10
atr_steps = 0.6
risk_ratio = 3
evaluation_period = 200
iterations = 100000
learning_rate = 0.01

columns = ['Close', 'Open', 'High', 'Low', 'S_trend', 'EMA3', 'EMA7', 'EMA20', 'SMA50',
           'RSI', 'ATR', 'S_trend_d', 'L', 'U', 'MACD', 'MACD_hist', 'MACD_signal', 'CTI',
           'WMA9', 'WMA14', 't1', 't2', 't3']

model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, loss_function='Logloss', depth=10)
# model = LogisticRegression()


def parse_data(ticker='MSFT'):
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

    return df


def get_data(ticker='MSFT'):

    df = parse_data(ticker)

    df['target'] = 0

    for index, row in df[:-bet_period].iterrows():
        result = 0

        current_price = df['Close'][index]
        # stop_loss = df['Close'][index] - atr_steps * df['ATR'][index]
        # take_profit = df['Close'][index] + atr_steps * risk_ratio * df['ATR'][index]

        # print(f'Price {current_price} {stop_loss} {take_profit}')

        # if current_price > 1:  # close price higher that EMA 200
        for i in range(index + 1, index + bet_period):
            if df['Low'][i] <= current_price * stop_loss:
                result = 0
                break
            elif df['High'][i] >= current_price * take_profit:
                result = 1
                break

        df.at[index, 'target'] = result

    df = df[200:-bet_period].copy()  # we remove latest "bet_period" days because we don't know the results of bets

    return df


def train_model(df):
    df_train = df[:-evaluation_period].copy()  # make sure we do not take the latest data to the evaluation scope

    X = df_train[columns]
    y = df_train['target']

    print(X.shape[0], y.sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)


def evaluate_model(df):
    X = df[columns]
    y = df['target']

    X_test = X[-evaluation_period:]
    y_test = y[-evaluation_period:]

    # generate predictions on the test data
    y_pred = model.predict(X_test)

    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # print the evaluation metrics
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1-score: {f1:.3f}')
        print(f'ROC AUC: {roc_auc:.3f}')

        return accuracy, precision, f1
    except Exception as e:
        print(e)
        exit(1)
        pass


def get_prediction(ticker='MSFT'):
    df = parse_data(ticker)

    history = df[-bet_period:]
    X = df[columns]
    X = X[-bet_period:]

    # generate predictions on the test data
    y_pred = model.predict(X)

    """
    for i, y in enumerate(y_pred):
        if y == 1:
            price = list(X['Close'])[i] * list(history['SMA200'])[i]
            print(f'Price: {price:.3f}, stop loss {price*stop_loss:.3f}, take profit: {price*take_profit:.3f}')
    """

    if y_pred[0] == 1:
        price = list(X['Close'])[0] * list(history['SMA200'])[0]
        print(f'{ticker}  Price: {price:.3f}, stop loss {price * stop_loss:.3f}, take profit: {price * take_profit:.3f}')

        order_stop_loss = price * stop_loss
        order_take_profit = price * take_profit

        for i in range(1, bet_period - 1):
            low = list(X['Low'])[i] * list(history['SMA200'])[i]
            high = list(X['High'])[i] * list(history['SMA200'])[i]

            if low < order_stop_loss:
                print('FAIL')
                break
            elif high > order_take_profit:
                print('PROFIT')
                break

        print('*)')

    # print(y_pred)


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
TICKERS.remove('CEG')
TICKERS.remove('ELV')

ticker = 'MSFT'
best_f1 = 0

df2 = get_data(ticker)
train_model(df2)
accuracy, precision, f1 = evaluate_model(df2)

# print(df2.tail(30))
# print(df2.columns)

for i in range(10):
    print(df2['Date'][i])

if f1 > best_f1:
    best_f1 = f1

    print(' * ' * 20)
    print('PARAMS:')
    print(f"learning_rate={learning_rate}, take_profit={take_profit}, stop_loss={stop_loss}, bet_period={bet_period}, iterations={iterations}")
    print(' * ' * 20)

"""
for ticker in TICKERS[:100]:
    print('-' * 20)
    print(ticker)

    df2 = get_data(ticker)
    train_model(df2)
    evaluate_model(df2)

    # get_prediction(ticker)
    # exit(1)
"""