
import numpy as np
import pandas as pd
from os import path
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly

from datetime import datetime
from utils import suppress_stdout_stderr


plt.style.use("fivethirtyeight")


data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])

TICKERS = set([t[0] for t in data.columns.values])

for ticker in TICKERS:
    ticker_data = data[ticker]['Close']
    # Log2 from real price to make it smooth
    # ticker_data = np.log(ticker_data['Close'])

    model = Prophet(changepoint_prior_scale=0.5)
    model.add_country_holidays(country_name='US')

    df = ticker_data.reset_index()
    df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

    with suppress_stdout_stderr():
        model.fit(df)

    future = model.make_future_dataframe(periods=30)
    # Remove weekends from prediction, because we don't have data for weekends:
    future = future[future['ds'].dt.weekday <= 4]
    forecast = model.predict(future)

    res = plot_plotly(model, forecast)
    res.update_layout(height=1000, width=1500)
    res.show()

    exit(1)
