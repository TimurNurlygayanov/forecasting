# After the model was trained on specific ticker, it shows very good results for this ticker
# but if it was trained on a different ticker, it doesn't show good results.

from catboost import CatBoostClassifier
import pandas as pd
import pandas_ta  # for TA magic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings


# Set the warnings filter to "ignore"
warnings.filterwarnings("ignore")

stop_loss = 0.95
take_profit = 1.1
bet_period = 30 * 8
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, loss_function='Logloss')


def get_data(ticker='MSFT'):
    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period='700d', interval='1h')
    df = df.reset_index()

    df.ta.ema(length=3, append=True, col_names=('EMA3',))
    df.ta.ema(length=7, append=True, col_names=('EMA7',))
    df.ta.ema(length=20, append=True, col_names=('EMA20',))

    df.ta.sma(length=50, append=True, col_names=('SMA50',))
    df.ta.sma(length=200, append=True, col_names=('SMA200',))

    df.ta.rsi(append=True, col_names=('RSI',))
    df.ta.atr(append=True, col_names=('ATR',))
    df.ta.supertrend(append=True, length=8*10, multiplier=4.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
    df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)

    for x in ['Close', 'Open', 'High', 'Low', 'EMA3', 'EMA7', 'EMA20', 'SMA50', 'L', 'U']:
        df[x] /= df['SMA200']

    df['target'] = 0

    for index, row in df[:-bet_period].iterrows():
        result = 0

        current_price = df['Close'][index]

        for i in range(index + 1, index + bet_period):
            if df['Low'][i] <= current_price * stop_loss:
                result = 0
                break
            elif df['High'][i] >= current_price * take_profit:
                result = 1
                break

        df.at[index, 'target'] = result

    df = df[200:-bet_period].copy()

    return df


def train_model(df):
    X = df[['Close', 'Open', 'High', 'Low', 'EMA3', 'EMA7', 'EMA20', 'SMA50',
            'RSI', 'ATR', 'S_trend_d', 'L', 'U']]
    y = df['target']

    # print(X.shape[0], y.sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)


def evaluate_model(df):
    X = df[['Close', 'Open', 'High', 'Low', 'EMA3', 'EMA7', 'EMA20', 'SMA50', 'SMA200',
            'RSI', 'ATR', 'S_trend_d', 'L', 'U']]
    y = df['target']

    # print(X.shape[0], y.sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # generate predictions on the test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # print the evaluation metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('ROC AUC:', roc_auc)


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
TICKERS.remove('CEG')
TICKERS.remove('ELV')


for ticker in TICKERS[:20]:
    print('-' * 20)
    print(ticker)

    df2 = get_data(ticker)
    train_model(df2)

for ticker in TICKERS[:3]:
    print('-' * 20)
    print(ticker)

    df2 = get_data(ticker)
    evaluate_model(df2)
