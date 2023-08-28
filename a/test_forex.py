# This script predicts 1:4 shares tranding

from bot.utils import get_data
from bot.utils import get_tickers_polygon
from bot.utils import get_ticker_details

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from catboost import Pool
from catboost import CatBoostClassifier

from tqdm import tqdm


def is_engulfing_bullish(open1, open2, close1, close2):
    if open1 > close1:   # previous candle is red
        if open2 < close2:  # current candle is green
            if open2 < close1 and close2 > open1:  # green is bigger
                return 1

    return 0


# TICKERS = get_tickers_polygon(limit=1000)  # 2000
TICKERS = ['C:USDJPY']
new_matrix = []
results = []
step_size = 3  # make sure we do not use similar data for training and verification

for ticker in tqdm(TICKERS):
    df = get_data(ticker, period='minute')

    exit(1)

    if df is None:
        continue

    df = df[200:]

    for i, (index, row) in enumerate(df.iterrows()):
        # Only take every 10 measurement
        if i % step_size != 0:
            continue

        state = [0] * 98
        result = 0

        if len(df) - 40 > i > 200:
            for j in range(10):     # 50
                close_x = df['Close'].values[i - j]
                open_x = df['Open'].values[i - j]

                if close_x > open_x:
                    state[j] = 1   # is this a green candle?

            if sum(state[7:10]) == 3:
                state[10] = 1  # last 3 candles are green?

            state[11] = is_engulfing_bullish(
                df['Open'].values[i - 1], df['Open'].values[i],
                df['Close'].values[i - 1], df['Close'].values[i]
            )

            state[12] = 1 if row['RSI'] < 30 else 0
            state[13] = 1 if row['RSI'] < 40 else 0
            state[14] = 1 if row['RSI'] > 70 else 0
            state[15] = 1 if row['S_trend_d'] > 0 else 0
            state[16] = 1 if row['S_trend_d'] > 0 > df['S_trend_d'].values[i - 1] else 0

            state[17] = 1 if row['Low'] > row['EMA200'] else 0
            state[18] = 1 if row['Low'] > row['EMA50'] else 0
            state[19] = 1 if row['EMA50'] > row['EMA200'] else 0

            state[20] = 1 if row['Close'] > row['EMA200'] else 0
            state[21] = 1 if row['Close'] > row['EMA50'] else 0

            # lower low and higher high
            state[22] = 1 if row['Low'] < min(df['Low'].values[i-10:i-1]) else 0
            state[23] = 1 if row['Low'] < min(df['Low'].values[i - 50:i - 1]) else 0
            state[24] = 1 if row['Low'] < min(df['Low'].values[i - 200:i - 1]) else 0

            state[25] = 1 if row['High'] > max(df['High'].values[i - 10:i - 1]) else 0
            state[26] = 1 if row['High'] > max(df['High'].values[i - 50:i - 1]) else 0
            state[27] = 1 if row['High'] > max(df['High'].values[i - 200:i - 1]) else 0

            # if price higher that EMA for long time?
            higher_price = True
            for j in range(10):
                if df['Close'].values[i-j] < df['EMA50'].values[i-j]:
                    higher_price = False
            state[28] = 1 if higher_price else 0

            higher_price = True
            for j in range(10):
                if df['Close'].values[i - j] < df['EMA200'].values[i - j]:
                    higher_price = False
            state[29] = 1 if higher_price else 0

            state[30] = 1 if df['EMA200'].values[i] > df['EMA200'].values[i - 10] else 0
            state[31] = 1 if df['EMA50'].values[i] > df['EMA50'].values[i - 10] else 0

            state[32] = 1 if row['MACD_hist'] > df['MACD_hist'].values[i - 1] > df['MACD_hist'].values[i - 2] else 0
            state[33] = 1 if row['MACD_hist'] > 0 else 0

            candle_full = abs(row['High'] - row['Low'])
            candle_body = abs(row['High'] - row['Low'])
            green_hammer = row['Close'] > row['Open'] and (row['Open'] - row['Low']) / candle_full > 0.7
            state[34] = 1 if candle_full > candle_body * 3 and green_hammer else 0

            state[35] = 1 if df['volume'].values[i] > df['volume'].values[i - 1] else 0
            state[36] = 1 if df['volume'].values[i - 1] > df['volume'].values[i - 2] else 0
            state[37] = 1 if df['volume'].values[i - 2] > df['volume'].values[i - 3] else 0

            state[38] = 1 if df['vwap'].values[i] > df['vwap'].values[i - 1] else 0
            state[39] = 1 if df['vwap'].values[i - 1] > df['vwap'].values[i - 2] else 0
            state[40] = 1 if df['vwap'].values[i - 2] > df['vwap'].values[i - 3] else 0

            state[41] = 1 if row['High'] > row['U'] else 0
            state[42] = 1 if row['Close'] > row['U'] else 0

            state[43] = 1 if row['Low'] < row['L'] else 0
            state[44] = 1 if row['Close'] < row['L'] else 0

            state[45] = 1 if df['EMA7'].values[i] > df['EMA7'].values[i - 1] else 0
            state[46] = 1 if df['Close'].values[i] > df['EMA7'].values[i] else 0

            state[47] = 1 if row['S_trend_d34'] > 0 else 0
            state[48] = 1 if row['S_trend_d34'] > 0 > df['S_trend_d34'].values[i - 1] else 0

            state[49] = 1 if row['CDL_DOJI_10_0.1'] > 0 else 0
            state[50] = 1 if row['CDL_MORNINGSTAR'] > 0 else 0
            state[51] = 1 if row['CDL_HAMMER'] > 0 else 0
            state[52] = 1 if row['CDL_SHOOTINGSTAR'] > 0 else 0
            state[53] = 1 if row['CDL_ENGULFING'] > 0 else 0
            state[54] = 1 if row['CDL_ENGULFING'] < 0 else 0

            if df['EMA50'].values[i] > df['EMA200'].values[i]:
                if df['EMA50'].values[i-1] < df['EMA200'].values[i-1]:
                    state[55] = 1

            if df['EMA7'].values[i] > df['EMA50'].values[i]:
                if df['EMA7'].values[i - 1] < df['EMA50'].values[i - 1]:
                    state[56] = 1

            max_profit = (max(df['High'].values[i+1:i+step_size]) - row['Close']) / row['Close']
            max_lose = (min(df['Low'].values[i+1:i+step_size]) - row['Close']) / row['Close']
            average_profit = (sum(df['Low'].values[i+1:i+step_size]) / step_size - row['Close']) / row['Close']

            # high precision, but small amount of predictions
            # if 0.05 > max_lose > 0 and max_profit / max_lose > 2 and max_profit > 0.1:
            # if max_lose == 0:
            #     max_lose = -0.0001

            # 2&6% gives more deals, 2&10 gives too small
            # if max_lose > -0.02 and max_profit > 0.08:

            # print(max_profit*100, max_lose * 100)

            if max_profit > abs(max_lose) * 2:
                result = 1
                print(max_profit * 100)

            new_matrix.append(state)
            results.append(result)


def evaluate_model(X_test, y_test):
    y_pred = model.predict(X_test)

    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        print('Total deals predicted ', sum(y_pred), 100*sum(y_pred)/len(y_pred))

        # print the evaluation metrics
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1-score: {f1:.3f}')
        print(f'ROC AUC: {roc_auc:.3f}')

        return accuracy, precision, f1
    except Exception as e:
        print(e)
        exit(1)
        pass


# model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
model = CatBoostClassifier(iterations=10000, depth=10, thread_count=7, learning_rate=0.001, loss_function='Logloss')
X_train, X_test, y_train, y_test = train_test_split(new_matrix, results, test_size=0.2, random_state=42)

# sample_weights = [1 if y == 1 else 0.9 for y in y_train]
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)  # sample_weight=sample_weights

evaluate_model(X_test, y_test)
