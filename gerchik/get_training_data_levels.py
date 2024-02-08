import uuid

from bot.utils import get_data
from bot.utils import get_tickers_polygon

import numpy as np
from scipy.signal import argrelextrema

from joblib import Parallel, delayed

from uuid import uuid4

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from catboost import CatBoostClassifier

from utils import draw
from utils import calculate_atr
from utils import find_nakoplenie


TICKERS = get_tickers_polygon(limit=5000)  # 2000
RESULTS = []
BAD_RESULTS = []
TICKERS = TICKERS[:200]


def check_for_bad_candles(df, atr):
    gaps = 0

    for i, (index, row) in enumerate(df.iterrows()):
        if i > 0:
            if abs(row['High'] - row['Low']) < 0.4 * atr:
                gaps += 1
            if abs(row['High'] - row['Low']) < 0.1:
                gaps += 1

    if gaps / len(df) > 0.1:
        return True

    return False


def search_for_bsu(lows, highs, bsu_price, luft):
    for i in range(len(lows) - 3):
        if abs(lows[i] - bsu_price) < 0.1 * luft:
            return True
        if abs(highs[i] - bsu_price) < 0.1 * luft:
            return True

    return False


def check_dozhatie(df):
    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    opens = df['Open'].tolist()
    closes = df['Close'].tolist()
    s1 = highs[-1] - lows[-1]
    s2 = highs[-2] - lows[-2]
    s3 = highs[-3] - lows[-3]

    delta = 0.03

    if s1 < s2 < s3:   # volatilnost padaet
        if lows[-1] > lows[-2] > lows[-3]:
            if opens[-1] < closes[-1]:
                # Check for the confirmation
                k = 0
                for high in highs[-10:-1]:
                    if abs(high - highs[-1]) <= delta:
                        k += 1

                if k >= 2:
                    return highs[-1]  # draw(df, file_name=ticker, ticker=ticker, level=highs[-1])

        if highs[-1] < highs[-2] < highs[-3]:
            if opens[-1] > closes[-1]:
                k = 0
                for low in lows[-10:-1]:
                    if abs(low - lows[-1]) <= delta:
                        k += 1

                if k >= 2:
                    return lows[-1]  # draw(df, file_name=ticker, ticker=ticker, level=lows[-1])

    return 0


def check_scenario(df, level):
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    current_close = df['Close'].tolist()[-1]

    last_candle_size = highs[-1] - lows[-1]
    ratio = abs(level - current_close) / last_candle_size

    blizhnii_retest = False
    for i in range(4, 12):
        if lows[-i] < level < highs[-i]:
            blizhnii_retest = True

    label = f'Ratio: {round(ratio, 2):.2f}'
    if blizhnii_retest:
        label += '<br> Ближний ретест'
    else:
        label += '<br> Дальний ретест'

    return {
        'x0': df.index[-3], 'x1': df.index[-1], 'y0': min(lows[-3:]), 'y1': max(highs[-3:]),
        'label': label, 'color': 'rgba(55,200,34,0.2)'
    }


def run_me(ticker):
    global RESULTS
    global BAD_RESULTS

    diff_days = 10

    dfX = get_data(ticker, period='day', days=400, save_data=False)
    dfX.index = dfX.index.strftime('%b %d')

    if dfX is None or dfX.empty or len(dfX) < 20:
        return None

    average_volume = (sum(dfX['volume'].tolist()) / len(dfX)) // 1000
    if average_volume < 300:  # only take shares with 1M+ average volume
        return None

    current_price = dfX['Close'].tolist()[-1]
    if current_price < 1:
        return None  # ignore penny stocks and huge stocks

    atr = calculate_atr(dfX)
    if check_for_bad_candles(dfX, atr):
        return None

    min_index = 200  # start collecting data only if we have data for at least 200 days
    max_index = len(dfX) - 6  # 5 days to get results

    if len(dfX) < min_index:
        return None

    for start_position in range(min_index, max_index, 1):
        df = dfX.iloc[start_position-200:start_position+1].copy()

        lows = df['Low'].tolist()
        highs = df['High'].tolist()
        open_prices = df['Open'].tolist()
        close_prices = df['Close'].tolist()

        luft = 0.02 * atr

        found_signal = False
        levels = []

        # paranormal bars level search:

        for i, (index, row) in enumerate(df.iterrows()):
            bar_size = row['High'] - row['Low']

            if i < len(df) - 5 and bar_size > 2 * atr:
                level_h = row['High']
                level_l = row['Low']

                for j in range(i+1, len(df) - 1):
                    if df['Open'].iloc[j] < level_h < df['Close'].iloc[j]:
                        level_h = 0
                    if df['Open'].iloc[j] > level_h > df['Close'].iloc[j]:
                        level_h = 0

                    if df['Open'].iloc[j] < level_l < df['Close'].iloc[j]:
                        level_l = 0
                    if df['Open'].iloc[j] > level_l > df['Close'].iloc[j]:
                        level_l = 0

                if level_h:
                    levels.append(level_h)
                if level_l:
                    levels.append(level_l)

        if abs(highs[-1] - highs[-2]) < 0.02 or abs(highs[-1] - highs[-3]) < 0.02 or abs(highs[-1] - highs[-4]) < 0.02:
            levels.append(highs[-1])
        if abs(lows[-1] - lows[-2]) < 0.02 or abs(lows[-1] - lows[-3]) < 0.02 or abs(lows[-1] - lows[-4]) < 0.02:
            levels.append(lows[-1])

        #  limit + mirror levels search

        prices = sorted(highs + lows)
        bars_required = 3

        group = []
        previous_price = prices[0]

        for p in prices:
            if 100 * abs(previous_price - p) / p < 0.5 * luft:
                group.append(p)
            else:
                if len(group) >= bars_required:
                    level = min(group)

                    levels.append(level)

                group = []

            previous_price = p

        # izlom trenda search

        border = 10

        # Find local minima and maxima indices
        minima_idx = argrelextrema(df['Low'].values, np.less, order=border)[0]
        maxima_idx = argrelextrema(df['High'].values, np.greater, order=border)[0]

        # Get corresponding data points
        local_minima = df.iloc[minima_idx]
        local_maxima = df.iloc[maxima_idx]

        for i, (index, row) in enumerate(local_minima.iterrows()):
            levels.append(row['Low'])

        for i, (index, row) in enumerate(local_maxima.iterrows()):
            levels.append(row['High'])

        level = check_dozhatie(df)
        if level > 0:
            levels.append(level)

        # Choosing the right level:

        found_signal = False
        selected_level = 0
        for level in levels:
            # Check if level is clear:

            k = 0
            for i in range(0, 30):
                if lows[-i] < level < highs[-i]:
                    k += 1

            if k < 2:
                # Then check if we have some touch of the level
                if abs(current_price - level) < 3 * luft:
                    found_signal = True
                    selected_level = level
                else:
                    if abs(lows[-1] - level) < luft:
                        found_signal = True
                        selected_level = level

                    if abs(highs[-1] - level) < luft:
                        found_signal = True
                        selected_level = level

        if found_signal:
            result = dfX.iloc[start_position+1:start_position+6]
            res_low = min(result['Low'].tolist())
            res_high = max(result['High'].tolist())

            df_ref = df.iloc[-10:].copy()
            min_low = min(df_ref['Low'].tolist())  # it is new 0
            max_high = max(df_ref['High'].tolist())  # it is new 1

            df_ref['Close'] -= min_low
            df_ref['Low'] -= min_low
            df_ref['Open'] -= min_low
            df_ref['High'] -= min_low

            df_ref['Close'] /= (max_high - min_low)
            df_ref['Low'] /= (max_high - min_low)
            df_ref['Open'] /= (max_high - min_low)
            df_ref['High'] /= (max_high - min_low)

            r = df_ref['Open'].tolist() + df_ref['Close'].tolist() + df_ref['Low'].tolist() + df_ref['High'].tolist()
            level_value = (selected_level - min_low) / (max_high - min_low)
            r.append(level_value)

            res = 'stop'
            close_price = df['Close'].tolist()[-1]

            if abs(res_low / close_price) > 0.06 and abs(res_high / close_price) < 0.02:
                res = 'short'
            if abs(res_low / close_price) < 0.02 and abs(res_high / close_price) > 0.06:
                res = 'long'

            if res_low >= selected_level and abs(selected_level-close_price) / close_price > 0.10:
                res = 'long from level'
                draw(
                    df,
                    level=selected_level, file_name=str(uuid4()), ticker=ticker + ' long from level', future=5
                )
            if res_high <= selected_level and abs(selected_level-close_price) / close_price > 0.10:
                res = 'short from level'
                draw(
                    df,
                    level=selected_level, file_name=str(uuid4()), ticker=ticker + ' short from level', future=5
                )

            RESULTS.append({'data': r, 'result': res, 'data_raw': df})


def evaluate_model(model_x, X_test, y_test):
    y_pred = model_x.predict(X_test)

    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(y_test, y_pred, average='macro')

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


print('Starting threads...')
Parallel(n_jobs=10, require='sharedmem', timeout=200)(delayed(run_me)(ticker) for ticker in TICKERS)

new_matrix = []
item_results = []
preview = {k: 0 for k in ['stop', 'long', 'short', 'long from level', 'short from level']}
for r in RESULTS:
    new_matrix.append(r['data'])
    item_results.append(r['result'])

    preview[r['result']] += 1


print(f'We have {len(RESULTS)} samples')
print(preview)

# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(iterations=1000, depth=10, thread_count=7, learning_rate=0.001, loss_function='MultiClass')  # loss_function='Logloss'
X_train, X_test, y_train, y_test = train_test_split(new_matrix, item_results, test_size=0.1, random_state=42)

print(f'Positive options in train dataset: {len(y_train)} / {len(X_train)}')
print(f'Positive options in evaluation dataset: {len(y_test)} / {len(X_test)}')

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

evaluate_model(model, X_test, y_test)
