# This script predicts 2:8 shares trading (2k shares):
# Accuracy: 0.864
# Precision: 0.867
# Recall: 0.011
# F1-score: 0.021
# ROC AUC: 0.505

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

from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles


TICKERS = get_tickers_polygon(limit=5000)  # 2000
# TICKERS = get_tickers()  # ONLY S&P500

new_matrix = []
latest_state = {}
results = []
step_size = 3  # make sure we do not use similar data for training and verification
max_days = 600
risk_reward_ratio = 3

TICKERS = TICKERS[:100]

for ticker in tqdm(TICKERS):
    df = get_data(ticker, period='day', days=max_days)

    if df is None or len(df) < 201:
        continue

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        continue

    current_price = df['Close'].tolist()[-1]
    if current_price < 1:
        continue  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        continue

    # df = df[200:]

    for i, (index, row) in enumerate(df.iterrows()):
        # Only take every K measurement
        if i % step_size != 0:
            continue

        result = 0

        if 200 < i < len(df) - 10:
            state = get_state(df, i=i, step_size=step_size)

            # we only check for long positions with risk reward ratio = 1:5
            buy_price = row['Close']
            stop_loss = row['EMA21_low']
            take_profit = row['Close'] + risk_reward_ratio * abs(row['Close'] - row['EMA21_low'])

            deal_is_done = False
            for j in range(i+1, min(len(df)-1, i + 50)):
                if not deal_is_done:
                    if df['Low'].values[j] <= stop_loss:
                        result = 0
                        deal_is_done = True
                    elif df['High'].values[j] >= take_profit:
                        result = 1
                        deal_is_done = True
            #

            new_matrix.append(state)
            results.append(result)

    latest_state[ticker] = get_state(df, len(df)-1)


def evaluate_model(model_x, X_test, y_test):
    # y_pred = model_x.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)      # 0.9 >  this increases precision from 0.28 to 0.33

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


print(f'TOTAL DATA {len(results)}')
print(f'POSITIVE CASES: {sum(results)}')

# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(
    iterations=1000, depth=10, thread_count=7,
    learning_rate=0.1, loss_function='Logloss',
    class_weights=[1, 1.2]
)
X_train, X_test, y_train, y_test = train_test_split(new_matrix, results, test_size=0.1, random_state=42)

print(f'Positive options in train dataset: {sum(y_train)} / {len(X_train)}')
print(f'Positive options in evaluation dataset: {sum(y_test)} / {len(X_test)}')

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f'Positive options in balanced train dataset: {sum(y_train_balanced)} / {len(X_train_balanced)}')

# sample_weights = [1 if y == 1 else 0.9 for y in y_train]
model.fit(X_train_balanced, y_train_balanced, eval_set=(X_test, y_test), verbose=False)  # sample_weight=sample_weights

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
