import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from joblib import Parallel, delayed

import json

from bot.utils import get_state

from gerchik.utils import calculate_atr
from gerchik.utils import check_for_bad_candles

from bot.utils import get_data
from bot.utils import get_tickers_polygon


TICKERS = get_tickers_polygon(limit=5000)  # 2000
RESULTS = []
STATES = []
max_days = 1000


def run_me(ticker, callback=None):
    global RESULTS

    callback()

    df = get_data(ticker, period='day', days=max_days)

    if df is None or len(df) < 320:
        return None

    average_volume = (sum(df['volume'].tolist()) / len(df)) // 1000
    if average_volume < 100:  # only take shares with 100k+ average volume
        return None

    current_price = df['Close'].tolist()[-1]
    if current_price < 1:  # this actually helps because penny stocks behave differently
        return None  # ignore penny stocks and huge stocks

    atr = calculate_atr(df)
    if check_for_bad_candles(df, atr):
        return None

    df_to_check = df.iloc[-120:].copy()

    prev_rsi = 0
    prev_50 = 0
    prev_200 = 0
    for i, (index, row) in enumerate(df_to_check.iterrows()):
        if 10 < i < len(df_to_check) - 20:
            """
            # RSI > 50
            # Failed: 258, passed: 72, win rate: 21.82%
            # Profit: 298.42%
            # RSI > 33 and risk:reward 1:5
            # Failed: 2814, passed: 360, win rate: 11.34%
            # Profit: 2415.50%
            if row['Close'] < row['EMA21']:
                # if row['EMA50'] > row['EMA200'] and row['RSI'] > 50 > prev_rsi:
                if row['RSI'] > 33 > prev_rsi:
                    buy_price = df_to_check['Open'].values[i+1]
                    if buy_price >= row['Close']:
                        stop_loss = row['Close']  # row['Low']
                        take_profit = buy_price + 5 * abs(row['Low'] - buy_price)
                        profit = 0
                        for j in range(i+1, len(df_to_check)):
                            if profit == 0:
                                if df_to_check['Low'].values[j] < stop_loss:
                                    profit = -abs(buy_price - stop_loss) / buy_price
                                elif df_to_check['High'].values[j] > take_profit:
                                    profit = abs(take_profit - buy_price) / buy_price

                        if profit != 0 and abs(stop_loss - buy_price) / buy_price < 0.03:
                            RESULTS.append(profit)

                            state = get_state(df_to_check, i)
                            STATES.append({'state': state[73:], 'result': 1 if profit > 0 else 0})
            """

            # EMA 50 & EMA200:
            # Failed: 224, passed: 83, win rate: 27.04%
            # Profit: 229.31%
            #
            # with EMA21 & EMA 50:
            # Failed: 316, passed: 262, win rate: 45.33%
            # Profit: 1036.64%
            if i > 5:
                if row['EMA21'] > row['EMA50'] and prev_50 < prev_200:
                    volatility = max(df_to_check['High'].values[i-5:i+1]) - min(df_to_check['Low'].values[i-5:i+1])

                    buy_price = df_to_check['Open'].values[i + 1]
                    stop_loss = buy_price - volatility / 2
                    take_profit = buy_price + 3 * abs(stop_loss - buy_price)
                    profit = 0

                    for j in range(i + 1, len(df_to_check)):
                        if profit == 0:
                            if df_to_check['Low'].values[j] < stop_loss:
                                profit = -abs(buy_price - stop_loss) / buy_price
                            elif df_to_check['High'].values[j] > take_profit:
                                profit = abs(take_profit - buy_price) / buy_price

                    if profit != 0 and abs(stop_loss - buy_price) / buy_price < 0.03:
                        RESULTS.append(profit)

            prev_50 = row['EMA21']
            prev_200 = row['EMA50']
            prev_rsi = row['RSI']

            """
            if row['S_trend_d'] > 0 > df_to_check['S_trend_d'].values[i-1]:
                buy_price = df_to_check['Open'].values[i + 1]
                stop_loss = row['S_trend']
                take_profit = buy_price + 3 * abs(stop_loss - buy_price)
                profit = 0
                for j in range(i + 1, len(df_to_check)):
                    if profit == 0:
                        if df_to_check['Low'].values[j] < stop_loss:
                            profit = -abs(buy_price - stop_loss) / buy_price
                        elif df_to_check['High'].values[j] > take_profit:
                            profit = abs(take_profit - buy_price) / buy_price

                if profit != 0:
                    RESULTS.append(profit)
            """

    # Save data only if all conditions are met
    file_name = f'collect_data/data/{ticker}.parquet'
    df.to_parquet(file_name)


print('Preparing training dataset...')
with tqdm(total=len(TICKERS)) as pbar:
    def update():
        pbar.update()

    Parallel(n_jobs=-1, backend="threading", require='sharedmem', timeout=200)(
        delayed(run_me)(ticker, callback=update) for ticker in TICKERS
    )

failed = sum([1 for r in RESULTS if r < 0])
passed = sum([1 for r in RESULTS if r > 0])
profit_total = sum(RESULTS)

print(f'Failed: {failed}, passed: {passed}, win rate: {100 * passed / (passed+failed):.2f}%')
print(f'Profit: {100 * profit_total:.2f}%')

with open('collect_data/data.json', 'w+') as f:
    json.dump(STATES, f)
