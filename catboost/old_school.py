# !/usr/bin/python3
# -*- encoding=utf8 -*-

from catboost import Pool
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np

from os import path
import yfinance as yf


with open('revolut_tickers.txt', 'r') as f:
    TICKERS = f.readlines()
TICKERS = [t.replace('\n', '') for t in TICKERS]
TICKERS = TICKERS[:100]

LONG_PERIOD = 10   # период с данными, которые отдаем в модель
SHORT_PERIOD = 10  # период за который ожидаем доход
BENEFIT_RATE = 1.1  # 10% in SHORT PERIOD time frame

PERIODS = []
LABELS = []
WEIGHTS = []
CHECK_PERIODS = []
CHECK_LABELS = []
CHECK_TICKER = []

ACTUAL_PERIODS = []
ACTUAL_LABELS = []
ACTUAL_TICKER = []

if path.exists('data.xlsx'):
    # Load the data from file, if the file exists
    data = pd.read_excel('data.xlsx', index_col=0, header=[0, 1])
else:
    data = yf.download(' '.join(TICKERS), period='5y',
                       group_by='ticker', interval='1d')
    data.to_excel('data.xlsx', index=True, header=True)


for ticker in TICKERS:
    dates = [d.strftime("%d %b %Y") for d in data[ticker]['Close'].index]
    dates_of_the_week = [d.weekday() for d in data[ticker]['Close'].index]

    df = data[ticker].reset_index()
    full_history = df['Close']

    # Moving average for 10 and 50 days:
    ma10 = data[ticker]['Close'].rolling(window=10).mean()
    ma50 = data[ticker]['Close'].rolling(window=50).mean()

    # Exponential moving average for 10 and 50 days:
    ema10 = data[ticker]['Close'].ewm(span=10, adjust=False).mean()
    ema50 = data[ticker]['Close'].ewm(span=50, adjust=False).mean()

    # Normalize all data:
    full_history = full_history / full_history.abs().max()
    ma10 = ma10 / ma10.abs().max()
    ma50 = ma50 / ma50.abs().max()
    ema10 = ema10 / ema10.abs().max()
    ema50 = ema50 / ema50.abs().max()

    if len(full_history) > LONG_PERIOD + 2 * SHORT_PERIOD + 1:
        iterations = len(full_history) - (LONG_PERIOD + 15 + SHORT_PERIOD)

        # Exclude first 15 days, because RSI for first days is 0:
        for i in range(15, iterations):
            # данные по которым делаем предсказание:
            long_history = full_history[i:i + LONG_PERIOD].values.tolist()
            # volume = df['Volume'][i:i + LONG_PERIOD].values.tolist()
            # volume = list(np.log(volume))
            # а это цены после предсказания:
            guess_period = full_history[i + LONG_PERIOD:i + LONG_PERIOD + SHORT_PERIOD].values.tolist()

            category = 'bad'
            # If the price increased more than BENEFIT_RATE % for short period:
            # mean = sum(guess_period) / len(guess_period)
            if max(guess_period) > long_history[-1] * BENEFIT_RATE:
                category = 'good'

            # Фичи: нормализованная цена за LONG_PERIOD + RSI за LONG_PERIOD
            #       + номер дня недели + объем торгуемых акций
            # PERIODS.append(
            #     [dates_of_the_week[i], ] + long_history + volume + ma3[i:i + LONG_PERIOD] + ma21[i:i + LONG_PERIOD])
            period_data = long_history + ema10[i:i + LONG_PERIOD] + ema50[i:i + LONG_PERIOD]
            PERIODS.append(period_data)
            LABELS.append(category)

        check_i = - LONG_PERIOD - SHORT_PERIOD
        long_history = full_history[check_i:check_i + LONG_PERIOD].values.tolist()
        # volume = df['Volume'][check_i:check_i + LONG_PERIOD].values.tolist()
        # volume = list(np.log(volume))
        # а это цены после предсказания:
        guess_period = full_history[check_i + LONG_PERIOD:].values.tolist()
        category = 'bad'
        # If the price increased more than 5% for two weeks:
        # mean = sum(guess_period) / len(guess_period)
        if max(guess_period) > long_history[-1] * BENEFIT_RATE:
            category = 'good'

        # CHECK_PERIODS.append([dates_of_the_week[check_i], ] + long_history + volume +
        #                      ma3[check_i:check_i+LONG_PERIOD] + ma3[check_i:check_i+LONG_PERIOD])
        period_data = long_history + ema10[check_i:check_i+LONG_PERIOD] + ema50[check_i:check_i+LONG_PERIOD]
        CHECK_PERIODS.append(period_data)
        CHECK_LABELS.append(category)

print('Good:', LABELS.count('good'))
print('Bad:', LABELS.count('bad'))

print(PERIODS[-1])

# Specify the training parameters:
model = CatBoostClassifier(iterations=1^6, thread_count=7,
                           loss_function='Logloss')
train_label = LABELS
train_data = PERIODS
data = Pool(data=train_data,
            label=train_label)
model.fit(data)

# Check the model on new data:
print('-' * 20)
print('Check on new data:')
true_positive = 0
false_negative = 0
false_positive = 0
correct_predictions = 0
for i, period in enumerate(CHECK_PERIODS):
    prediction = model.predict(period)
    prediction_accuracy = model.predict_proba(period)
    print("Predicted: {0} ({1}) , actual: {2}".format(prediction, prediction_accuracy, CHECK_LABELS[i]))

    if prediction == CHECK_LABELS[i]:
        if prediction == 'good':
            true_positive += 1

        correct_predictions += 1
    else:
        if prediction == 'bad':
            false_negative += 1
        elif prediction == 'good':
            false_positive += 1

print('Accuracy: {0}'.format(correct_predictions / len(CHECK_LABELS)))

if true_positive + false_negative:
    recall = true_positive / (true_positive + false_negative)
    print('Recall: {0}'.format(recall))

if true_positive + false_positive:
    precision = true_positive / (true_positive + false_positive)
    print('Precision: {0}'.format(precision))

# Check the model on train data:
print('-' * 20)
print('Check on train data:')
true_positive = 0
false_negative = 0
false_positive = 0
correct_predictions = 0
for i, period in enumerate(PERIODS):
    prediction = model.predict(period)

    if prediction == LABELS[i]:
        if prediction == 'good':
            true_positive += 1

        correct_predictions += 1
    else:
        if prediction == 'bad':
            false_negative += 1
        elif prediction == 'good':
            false_positive += 1

print('Accuracy: {0}'.format(correct_predictions / len(LABELS)))

if true_positive + false_negative:
    recall = true_positive / (true_positive + false_negative)
    print('Recall: {0}'.format(recall))

if true_positive + false_positive:
    precision = true_positive / (true_positive + false_positive)
    print('Precision: {0}'.format(precision))

print('Positive Accuracy {0}'.format(true_positive / len([l for l in LABELS if l == 'good'])))