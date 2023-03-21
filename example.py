
import pandas as pd
import pandas_ta  # for TA magic
import matplotlib.pyplot as plt


# data = pd.read_csv('multiTimeline.csv')
data = pd.DataFrame()
data = data.ta.ticker("META", period='700d', interval='1h', auto_adjust=True)


# Use power of Pandas TA:
data.ta.ema(close=data['Close'], length=30, append=True, col_names=('EMA30',))
data.ta.ema(close=data['Close'], length=200, append=True, col_names=('EMA200',))
data.ta.macd(col_names=('MACD', 'MACD_hist', 'MACD_signal'))

print(data.head())

plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close')
plt.plot(data['EMA30'], label='EMA30')
plt.plot(data['EMA200'], label='EMA200')
plt.legend()
plt.show()
