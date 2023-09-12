# This script predicts 2:8 shares trading (2k shares):
# Accuracy: 0.864
# Precision: 0.867
# Recall: 0.011
# F1-score: 0.021
# ROC AUC: 0.505

from bot.utils import get_data
from bot.utils import get_tickers
from bot.utils import get_tickers_polygon
from bot.utils import get_state
from bot.utils import get_news
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from catboost import CatBoostClassifier

from tqdm import tqdm


TICKERS = get_tickers_polygon(limit=5000)  # 2000
# TICKERS = get_tickers()  # ONLY S&P500

new_matrix = []
latest_state = {}
results = []
step_size = 5  # make sure we do not use similar data for training and verification

# TICKERS = TICKERS[:200]

for ticker in tqdm(TICKERS):
    df = get_data(ticker)
    news = []  # get_news(ticker)

    if df is None or len(df) < 401:
        continue

    df['News'] = 0
    for n in news:
        date = n['date'].strftime('%Y-%m-%d')
        if date in df.index:
            df.loc[date, 'News'] = 1

    df = df[200:]

    for i, (index, row) in enumerate(df.iterrows()):
        # Only take every K measurement
        if i % step_size != 0:
            continue

        result = 0

        if len(df) - 40 > i > 200:
            state = get_state(df, i, step_size=step_size)

            print(state)
            exit(1)

            max_profit = (max(df['High'].values[i+1:i+step_size]) - row['Close']) / row['Close']
            max_lose = (min(df['Low'].values[i+1:i+step_size]) - row['Close']) / row['Close']

            # 2&6% gives more deals, 2&10 gives too small
            # if max_lose > -0.02 and max_profit > 0.08:
            if max_lose > -0.02 and max_profit > 0.06:
                result = 1

            new_matrix.append(state)
            results.append(result)

    latest_state[ticker] = get_state(df, len(df)-1)


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
              f'this is about {len(TICKERS) * sum(y_pred)/len(y_pred):.2f} deals / day')

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

# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(iterations=1000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
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
