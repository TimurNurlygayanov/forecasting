# Example of MACD + RSI + Super Trend indicators
#

import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
from plotly.subplots import make_subplots


ticker = 'A'
df = pd.DataFrame()
df = df.ta.ticker(ticker, period='700d')

# Calculation:
df.ta.rsi(length=14, append=True, col_names=('RSI', ))
df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))
df.ta.supertrend(length=10, multiplier=4.0, append=True,
                 col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))

df = df[-200:]

# Drawing:
graph = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.2, 0.2])
graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

# Price candlesticks:
graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], row=1, col=1)

# MACD:
graph.add_scatter(y=df['MACD'], mode='lines', name='MACD',
                  line={'color': 'black'}, row=2, col=1)
graph.add_scatter(y=df['MACD_signal'], mode='lines', name='MACD_signal',
                  line={'color': 'red'}, row=2, col=1)

# RSI
graph.add_scatter(y=df['RSI'], mode='lines', name='RSI',
                  line={'color': 'magenta'}, row=3, col=1)

# Super Trend:
graph.add_scatter(y=df['S_trend'], mode='lines', name='super_trend', row=1, col=1)
graph.add_scatter(y=df['S_trend_s'], mode='lines', name='s_trend_down',
                  line={'color': '#ff4040', 'width': 3}, row=1, col=1)
graph.add_scatter(y=df['S_trend_l'], mode='lines', name='s_trend_up',
                  line={'color': '#00ff7f', 'width': 3}, row=1, col=1)


graph.show()
