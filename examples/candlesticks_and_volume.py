# Example of candlesticks graph + volume
#

import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
from plotly.subplots import make_subplots


ticker = 'A'
df = pd.DataFrame()
df = df.ta.ticker(ticker, period='200d')

graph = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.4])
graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], row=1, col=1)

bar_colors = np.array(['#bada55'] * len(df))
bar_colors[df['Volume'] < df['Volume'].shift(1)] = '#ff4040'

graph.add_bar(y=df['Volume'], name='Volume', row=2, col=1, marker={'color': bar_colors})

graph.show()
