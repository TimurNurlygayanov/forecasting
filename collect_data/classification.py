import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE


df = pd.read_parquet('collect_data/data.parquet')

train_X = []
train_y = []

for i, (index, row) in enumerate(df.iterrows()):
  if row['result'] in ['good', 'bad']:
    r = []
    for p in df.columns.tolist():
      if p != 'result':
        value = 0 if np.isnan(row[p]) else row[p]
        r.append(value)

    if not any(np.isnan(x) for x in r):
      train_X.append(r)
      train_y.append(1 if row['result'] == 'good' else 0)



model = CatBoostClassifier(
    iterations=2000, use_best_model=True, depth=10,
    learning_rate=0.1, loss_function='Logloss', eval_metric='Precision',
    class_weights=[1, 1],   # this one helps to increase Recoll for the white dots
    custom_loss=['AUC', 'Precision'],
    # l2_leaf_reg=4,
    # bagging_temperature=0.1,
    # random_strength=0.2,
    # leaf_estimation_method='Newton',
)
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.1, random_state=42)

print(f'Positive options in train dataset: {sum(y_train)} / {len(X_train)}')
print(f'Positive options in evaluation dataset: {sum(y_test)} / {len(X_test)}')

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f'Positive options in balanced train dataset: {sum(y_train_balanced)} / {len(X_train_balanced)}')

model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=(X_test, y_test),
    early_stopping_rounds=200,
    verbose=10
)


def evaluate_model(model_x, X_test, y_test):
    # y_pred = model_x.predict(X_test)
    y_pred_proba = model_x.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.8).astype(int)      # 0.9 >  increase this to increase precision for good bets

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


evaluate_model(model, X_test, y_test)


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
