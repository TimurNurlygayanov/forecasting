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

from gerchik.utils import draw

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

TICKERS = TICKERS[:100]


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
            buy_price = row['High']
            stop_loss = row['High'] - 0.2 * atr
            take_profit = buy_price + risk_reward_ratio * abs(buy_price - stop_loss)

            # zakritie pod low, Get rid of "bad" stop losses
            if abs(take_profit - buy_price) < abs(row['High'] - row['Low']):
                local_pool.append((state, 0))
            else:
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

##

# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(
    iterations=10000, depth=10, thread_count=9, use_best_model=True,
    learning_rate=0.001, loss_function='Logloss', eval_metric='Precision',
    class_weights=[1, 2],   # this one helps to increase Recoll for the white dots
    custom_loss=['AUC', 'Precision'],
    # l2_leaf_reg=4,
    # bagging_temperature=0.1,
    # random_strength=0.2,
    # leaf_estimation_method='Newton',
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

model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=(X_test, y_test),
    early_stopping_rounds=1000,
    verbose=10
)

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

for ticker in TICKERS:
    df = get_data(ticker, period='day', days=max_days)

    # print(ticker, len(df))

    if df is None or len(df) < 211:
        continue

    state = get_state(df, len(df) - 10)

    res = model.predict_proba(state)

    if res[1] > 0.5:
        buy_price = df['Close'].tolist()[-10]
        stop_loss = min(df['Low'].tolist()[-15:-9]) - 0.01
        take_profit = buy_price + risk_reward_ratio * abs(buy_price - stop_loss)

        draw(
            df.iloc[-70:].copy(), file_name=ticker, ticker=ticker + f" {res[1]:.2f}",
            level=0, boxes=[], stop_loss=stop_loss, take_profit=take_profit
        )
