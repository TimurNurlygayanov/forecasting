# Example of Bollinger Bands graphs
#

import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
from plotly.subplots import make_subplots


ticker = 'msft'
df = pd.DataFrame()
df = df.ta.ticker(ticker, period='500d')

df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)

df.ta.ema(length=7, col_names=('EMA',), append=True)

df = df[-100:]

# Draw:
graph = make_subplots(rows=1, cols=1)
graph.update_layout(title=ticker, xaxis_rangeslider_visible=False)

graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], row=1, col=1)

graph.add_scatter(y=df['U'], mode='lines', name='Bollinger Upper',
                  line={'color': 'blue', 'width': 3})

graph.add_scatter(y=df['L'], mode='lines', name='Bollinger Low',
                  line={'color': 'red', 'width': 3})

graph.add_scatter(y=df['EMA'], mode='lines', name='EMA',
                  line={'color': 'magenta', 'width': 2})

graph.show()
