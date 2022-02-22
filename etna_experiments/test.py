from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.pipeline import Pipeline
import yfinance as yf
import plotly.graph_objects as go


# Choose a horizon
HORIZON = 3

data = yf.download('TSLA', period='2y',
                   group_by='ticker', interval='1d')

data = data['Close']
data = data.reset_index()
data["segment"] = "main"
data.rename(columns={'Close': 'target', 'Date': 'timestamp'}, inplace=True)
train_ts = data[:-HORIZON+1]
test_ts = data[-HORIZON:]
print(data.tail())

df = TSDataset.to_dataset(train_ts)
ts = TSDataset(df, freq="D")

print(ts)
print(dir(ts))

# Fit the pipeline
pipeline = Pipeline(model=ProphetModel(), horizon=HORIZON)
pipeline.fit(ts)

# Make the forecast
forecast_ts = pipeline.forecast()
forecast_ts = forecast_ts.to_pandas() # .reset_index()
forecast_ts = forecast_ts['main'].reset_index()
# print(forecast_ts['timestamp'])
print(forecast_ts)
# print(forecast_ts.to_pandas().reset_index())
print(test_ts.tail())
graph = go.Figure()
graph.add_scatter(x=test_ts['timestamp'], y=test_ts['target'],
                  name=f'TSLA Closed price - future real')
graph.add_scatter(x=train_ts['timestamp'], y=train_ts['target'],
                  name=f'TSLA Closed price - past')
graph.add_scatter(x=forecast_ts['timestamp'], y=forecast_ts['target'],
                  name=f'Prediction')
graph.show()
