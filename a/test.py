from bot.utils import get_data
from bot.utils import get_tickets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from catboost import Pool
from catboost import CatBoostClassifier


def is_engulfing_bullish(open1, open2, close1, close2):
    if open1 > close1:   # previous candle is red
        if open2 < close2:  # current candle is green
            if open2 < close1 and close2 > open1:  # green is bigger
                return 1

    return 0


TICKERS = get_tickets()
new_matrix = []
results = []

for ticker in TICKERS[:200]:
    df = get_data(ticker)

    if df is None:
        continue

    print(ticker)
    df = df[200:]

    for i, (index, row) in enumerate(df.iterrows()):
        state = [0] * 82
        result = 0

        if len(df) - 40 > i > 200:
            for j in range(50):
                close_x = df['Close'].values[i - j]
                open_x = df['Open'].values[i - j]

                if close_x > open_x:
                    state[j] = 1   # is this a green candle?

            if sum(state[47:50]) == 3:
                state[51] = 1  # last 3 candles are green?

            state[52] = is_engulfing_bullish(
                df['Open'].values[i - 1], df['Open'].values[i],
                df['Close'].values[i - 1], df['Close'].values[i]
            )

            state[53] = 1 if row['RSI'] < 30 else 0
            state[54] = 1 if row['RSI'] < 40 else 0
            state[55] = 1 if row['RSI'] > 70 else 0
            state[56] = 1 if row['S_trend_d'] > 0 else 0
            state[57] = 1 if row['S_trend_d'] > 0 > df['S_trend_d'].values[i - 1] else 0

            state[58] = 1 if row['Low'] > row['EMA200'] else 0
            state[59] = 1 if row['Low'] > row['EMA50'] else 0
            state[60] = 1 if row['EMA50'] > row['EMA200'] else 0

            state[61] = 1 if row['Close'] > row['EMA200'] else 0
            state[62] = 1 if row['Close'] > row['EMA50'] else 0

            # lower low and higher high
            state[63] = 1 if row['Low'] < min(df['Low'].values[i-10:i-1]) else 0
            state[64] = 1 if row['Low'] < min(df['Low'].values[i - 50:i - 1]) else 0
            state[65] = 1 if row['Low'] < min(df['Low'].values[i - 200:i - 1]) else 0

            state[66] = 1 if row['High'] > max(df['High'].values[i - 10:i - 1]) else 0
            state[67] = 1 if row['High'] > max(df['High'].values[i - 50:i - 1]) else 0
            state[68] = 1 if row['High'] > max(df['High'].values[i - 200:i - 1]) else 0

            # if price higher that EMA for long time?
            higher_price = True
            for j in range(10):
                if df['Close'].values[i-j] < df['EMA50'].values[i-j]:
                    higher_price = False
            state[69] = 1 if higher_price else 0

            higher_price = True
            for j in range(10):
                if df['Close'].values[i - j] < df['EMA200'].values[i - j]:
                    higher_price = False
            state[70] = 1 if higher_price else 0

            state[71] = 1 if df['EMA200'].values[i] > df['EMA200'].values[i - 10] else 0
            state[72] = 1 if df['EMA50'].values[i] > df['EMA50'].values[i - 10] else 0

            state[73] = 1 if row['MACD_hist'] > df['MACD_hist'].values[i - 1] > df['MACD_hist'].values[i - 2] else 0
            state[74] = 1 if row['MACD_hist'] > 0 else 0

            candle_full = abs(row['High'] - row['Low'])
            candle_body = abs(row['High'] - row['Low'])
            green_hammer = row['Close'] > row['Open'] and (row['Open'] - row['Low']) / candle_full > 0.7
            state[75] = 1 if candle_full > candle_body * 3 and green_hammer else 0

            state[76] = 1 if df['volume'].values[i] > df['volume'].values[i - 1] else 0
            state[77] = 1 if df['volume'].values[i - 1] > df['volume'].values[i - 2] else 0
            state[78] = 1 if df['volume'].values[i - 2] > df['volume'].values[i - 3] else 0

            state[79] = 1 if df['vwap'].values[i] > df['vwap'].values[i - 1] else 0
            state[80] = 1 if df['vwap'].values[i - 1] > df['vwap'].values[i - 2] else 0
            state[81] = 1 if df['vwap'].values[i - 2] > df['vwap'].values[i - 3] else 0

            max_profit = abs(max(df['High'].values[i+1:i+10]) - row['Close']) / row['Close']
            max_lose = abs(min(df['Low'].values[i+1:i+10]) - row['Close']) / row['Close']
            average_profit = (sum(df['Low'].values[i+1:i+10]) / 10 - row['Close']) / row['Close']

            if average_profit != 0:
                if max_lose == 0:
                    result = 1
                if max_lose > 0 and max_profit / max_lose > 2:
                    result = 1

                    # print(f'profit {100*max_profit:.2f}, lost: {100*max_lose:.2f}')

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

        print('Total deals predicted ', sum(y_pred))

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


model = CatBoostClassifier(iterations=10000, depth=6, thread_count=7, learning_rate=0.01, loss_function='Logloss')
X_train, X_test, y_train, y_test = train_test_split(new_matrix, results, test_size=0.2, random_state=42)

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

evaluate_model(X_test, y_test)
