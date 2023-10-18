# This script predicts 2:8 shares trading (2k shares):
# Accuracy: 0.864
# Precision: 0.867
# Recall: 0.011
# F1-score: 0.021
# ROC AUC: 0.505
import pandas as pd

from crypto_forex.utils import ALL_TICKERS
from crypto_forex.utils import get_data
from crypto_forex.utils import get_state

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from catboost import CatBoostClassifier

from tqdm import tqdm


new_matrix = []
latest_state = {}
results = []
step_size = 5  # make sure we do not use similar data for training and verification
medium_stop_loss = 0
STOP_LOSSES = {}

for ticker in tqdm(ALL_TICKERS):
    df = get_data(ticker, save_data=False, period='minute', multiplier=5)

    if df is None or len(df) < 401:
        continue

    dfHA = df.ta.ha()
    dfHA.rename(columns={'HA_open': 'Open', 'HA_close': 'Close', 'HA_low': 'Low', 'HA_high': 'High'}, inplace=True)

    df['green'] = df['Open'] < df['Close']
    df = df[200:]
    dfHA = dfHA[200:]

    df.index = pd.to_datetime(df.index, format='%Y-%m-%d, %H:%M:%S')

    for i, (index, row) in enumerate(df.iterrows()):
        # Only take every K measurement
        if i % step_size != 0:
            continue

        result = 0

        if len(df) - step_size * 2 > i > 200:
            state = get_state(df=df, dfHA=dfHA, i=i, step_size=step_size)

            max_profit = max(df['High'].values[i+1:i+step_size]) - row['Close']
            max_lose = min(df['Low'].values[i+1:i+step_size]) - row['Close']

            # print(max_profit, max_lose)

            # 2&6% gives more deals, 2&10 gives too small
            # if max_lose > -0.0002 and max_profit > 0.0008:
            # if max_lose >= 0 or abs(max_lose * 2) < max_profit:
            # if df['green'].values[i+1] and df['green'].values[i+2] and df['green'].values[i+3]:
            if 9 < index.hour < 22 and index.weekday() < 5:
                if max_lose >= 0 or abs(max_lose * 1.2) < max_profit:
                    result = 1

                new_matrix.append(state)
                results.append(result)

    latest_state[ticker] = get_state(df=df, dfHA=dfHA, i=len(df) - 1, step_size=step_size)
    latest_state[ticker] = get_state(df=df, dfHA=dfHA, i=len(df) - 2, step_size=step_size)
    latest_state[ticker] = get_state(df=df, dfHA=dfHA, i=len(df) - 3, step_size=step_size)


def evaluate_model(model_x, X_test, y_test):
    y_pred = model_x.predict(X_test)

    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Share total deals predicted and how much triggers algo will give every day
        print(f'Total deals predicted {sum(y_pred)}, '
              f'this is about {12 * sum(y_pred)/len(y_pred):.2f} deals / hour')

        # print the evaluation metrics
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1-score: {f1:.3f}')
        print(f'ROC AUC: {roc_auc:.3f}')

        return accuracy, precision, f1
    except Exception as e:
        print(e)
        pass


def custom_precision_loss(y_test, y_pred):
    # Replace this with your desired precision calculation.
    # You might use the precision formula or any other custom calculation.
    precision = precision_score(y_test, y_pred, zero_division=1)

    # You can return 1 - precision as the loss to be minimized.
    # The model will try to maximize precision.
    return 1 - precision


def custom_precision_metric(y_test, y_pred):
    precision = precision_score(y_test, y_pred, zero_division=1)
    return precision


model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
# model = CatBoostClassifier(iterations=1000, depth=4, thread_count=7, learning_rate=0.001, loss_function='Logloss')
# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001,
#                            loss_function=custom_precision_loss, custom_metric=custom_precision_metric)
X_train, X_test, y_train, y_test = train_test_split(new_matrix, results, test_size=0.1, random_state=42)

print(f'Positive options in train dataset: {sum(y_train)} / {len(X_train)}')
print(f'Positive options in evaluation dataset: {sum(y_test)} / {len(X_test)}')

# sample_weights = [1 if y == 1 else 0.9 for y in y_train]
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)  # sample_weight=sample_weights

evaluate_model(model, X_test, y_test)

print('* ' * 30)
for ticker, state in latest_state.items():
    res = model.predict(state)
    prediction_proba = model.predict_proba(state)

    if res == 1:
        print(ticker, res, prediction_proba)

# get_features_importance(model)
