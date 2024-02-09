# This script predicts 2:8 shares trading (2k shares):
# Accuracy: 0.864
# Precision: 0.867
# Recall: 0.011
# F1-score: 0.021
# ROC AUC: 0.505

import warnings
warnings.filterwarnings("ignore")

from bot.utils import get_data
from bot.utils import get_tickers_polygon
from bot.utils import get_state
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier

from tqdm import tqdm

from sklearn.cluster import KMeans
import sklearn.metrics as metrics

from joblib import Parallel, delayed

from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles


DATA_POOL = []
TICKERS = get_tickers_polygon(limit=5000)  # 2000

new_matrix = []
latest_state = {}
results = []
step_size = 2  # make sure we do not use similar data for training and verification
max_days = 1000
risk_reward_ratio = 5

# TICKERS = TICKERS[:100]

# Run 10 threads to get data to make this faster


def get_data_parallel(ticker, callback=None):
    get_data(ticker, period='day', days=max_days)

    callback()


print('Collecting data...')
with tqdm(total=len(TICKERS)) as pbar:
    def update():
        pbar.update()

    Parallel(n_jobs=-1, require='sharedmem', timeout=200)(
        delayed(get_data_parallel)(ticker, callback=update) for ticker in TICKERS
    )

# Data is collected, now we can use cached data in our loop


def run_me(ticker, callback=None):
    local_pool = []
    callback()

    df = get_data(ticker, period='day', days=max_days)

    if df is None or len(df) < 201:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return None

    current_price = df['Close'].tolist()[-1]
    if current_price < 10:  # this actually helps because penny stocks behave differently
        return None  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return None

    for i, (index, row) in enumerate(df.iterrows()):
        # Only take every K measurement
        if i % step_size != 0:
            continue

        result = 0

        if 200 < i < len(df) - 10:
            state = get_state(df, i=i)

            # we only check for long positions with risk reward ratio = 1:5
            buy_price = row['Close']
            stop_loss = row['Low'] - abs(row['Close'] - row['Low'])  # row['Low'] - abs(row['Close'] - row['Low'])  # row['Low'] - abs(row['Close'] - row['Low'])  #  0.98 * row['Close']  #  row['EMA21_low']  #  row['Low'] - abs(row['Close'] - row['Low'])  #  row['EMA21_low'])
            take_profit = buy_price + risk_reward_ratio * abs(buy_price - stop_loss)

            deal_is_done = False
            for j in range(i+1, min(len(df)-1, i + 10)):
                if not deal_is_done:
                    if df['Low'].values[j] <= stop_loss:
                        result = 0
                        deal_is_done = True
                    elif df['High'].values[j] >= take_profit:
                        result = 1
                        deal_is_done = True
            #

            # new_matrix.append(state)
            # results.append(result)
            local_pool.append((state, result))

    return local_pool, {'t': ticker, 's': get_state(df, len(df)-1)}


def evaluate_model(model_x, X_test, y_test):
    # y_pred = model_x.predict(X_test)
    y_pred_proba = model_x.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)      # 0.9 >  increase this to increase precision for good bets

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

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy, precision, f1
    except Exception as e:
        print(e)
        pass


print('Preparing training dataset...')
with tqdm(total=len(TICKERS)) as pbar:
    def update():
        pbar.update()

    _results = Parallel(n_jobs=-1, backend="threading", timeout=200)(delayed(run_me)(ticker, callback=update) for ticker in TICKERS)

    for r in _results:
        if r:
            DATA_POOL += r[0]
            latest_state[r[1]['t']] = r[1]['s']


for res in DATA_POOL:
    new_matrix.append(res[0])
    results.append(res[1])

print(f'TOTAL DATA {len(results)}')
print(f'POSITIVE CASES: {sum(results)}')

#
"""

kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(new_matrix)

clusters_win_rate = {k: {'total': 0, 'win': 0} for k in range(0, 20)}

for i, c in enumerate(clusters):
    clusters_win_rate[c]['total'] += 1
    clusters_win_rate[c]['win'] += results[i]

best_score = 0
best_class = {}
for c in clusters_win_rate.values():
    print(c)
    if c['win'] / c['total'] > best_score:
        best_score = c['win'] / c['total']
        best_class = c

print(f'BEST CLASS: {best_class}')
"""
#

# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(
    iterations=1000, depth=10, thread_count=7, #  task_type='GPU',
    learning_rate=0.1, eval_metric='Precision', loss_function='Logloss',
    class_weights=[1, 1]   # this one helps to increase Recoll for the white dots
)
X_train, X_test, y_train, y_test = train_test_split(new_matrix, results, test_size=0.1, random_state=42)

print(f'Positive options in train dataset: {sum(y_train)} / {len(X_train)}')
print(f'Positive options in evaluation dataset: {sum(y_test)} / {len(X_test)}')

# z = sum([1 for x in X_train if X_train.count(x) > 1])
# print(f'Total non unique values {z}')

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# X_train_balanced, y_train_balanced = X_train, y_train

print(f'Positive options in balanced train dataset: {sum(y_train_balanced)} / {len(X_train_balanced)}')

model.fit(X_train_balanced, y_train_balanced, eval_set=(X_test, y_test), verbose=False)

evaluate_model(model, X_test, y_test)

print('* ' * 30)
"""
for ticker, state in latest_state.items():
    res = model.predict(state)
    prediction_proba = model.predict_proba(state)

    if res == 1:
        print(ticker, res, prediction_proba)
"""

# get_features_importance(model)
