import pandas as pd

from crypto_forex.utils import ALL_TICKERS
from crypto_forex.utils import get_data


risk_reward_ratio = 1.5

df = get_data('C:EURUSD', save_data=False, period='minute', multiplier=15)
df.ta.ema(length=13, append=True, col_names=('EMA_short',))
df.ta.supertrend(append=True, length=11, multiplier=3.0,
                 col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
df.ta.atr(length=14, append=True, col_names=('ATR',))

# Filter data to get only data for trading days

df.index = pd.to_datetime(df.index, format='%Y-%m-%d, %H:%M:%S')

df = df[
    (df.index.hour >= 10) &
    (df.index.hour <= 22) &
    (df.index.weekday < 5)  # 0-4 represent Monday to Friday
]

total = 10000
deals = []
buy_position = 0
buy_price = 0
stop_loss = 0
take_profit = 0

for i, (index, row) in enumerate(df.iterrows()):
    if buy_position == 0:
        if 10 < index.hour < 21:
            if row['S_trend_d'] > 0 > df['S_trend_d'].values[i-1]:
                buy_price = row['Close']
                buy_position = i
                stop_loss = row['Low'] - row['ATR']
                take_profit = buy_price + risk_reward_ratio*(buy_price-stop_loss)

    if i > buy_position > 0:
        if row['Low'] < stop_loss:
            total -= 100000 * (buy_price-stop_loss) + 7
            buy_position = 0
            deals.append(0)
        elif row['High'] > take_profit:
            total += 100000 * (take_profit - buy_price) - 7
            buy_position = 0
            deals.append(1)

        if buy_position - i > 10:
            total += 100000 * (row['Close'] - buy_price) - 7
            deals.append(2)


failed = len([d for d in deals if d == 0]) or 1
success = len([d for d in deals if d == 1])
print('Failed deals: ', failed)
print(f'Success deals: {success} -> {100 * success / (success+failed):.2f}%')
print('Closed deals: ', len([d for d in deals if d == 2]))
print('Result: ', total)
