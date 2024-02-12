# This script predicts 2:8 shares trading (2k shares):
# Accuracy: 0.864
# Precision: 0.867
# Recall: 0.011
# F1-score: 0.021
# ROC AUC: 0.505

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier

from tqdm import tqdm
import json
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from gerchik.utils import draw

from joblib import Parallel, delayed


DATA_POOL = []

new_matrix = []
latest_state = {}
results = []


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
        print(f'Total deals predicted {sum(y_pred)}')

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
with open('collect_data/data.json', mode='r', encoding='utf-8') as f:
    DATA_POOL = json.load(f)

for res in DATA_POOL:
    state = [0 if np.isnan(x) else x for x in res['state']]
    new_matrix.append(state)
    results.append(res['result'])

print(f'TOTAL DATA {len(results)}')
print(f'POSITIVE CASES: {sum(results)}')

##

# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(
    iterations=10000, thread_count=9, use_best_model=True, depth=6,
    learning_rate=0.0001, loss_function='Logloss', eval_metric='Precision',
    class_weights=[1, 1],   # this one helps to increase Recoll for the white dots
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
    early_stopping_rounds=200,
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


# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_balanced, y_train_balanced)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)
print("Classification Report:")
print(classification_rep)