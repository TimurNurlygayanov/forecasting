from catboost import Pool
from catboost import CatBoostClassifier

import yfinance as yf
# import plotly.graph_objects as go


prediction_period = 14
working_days = 5 * prediction_period // 7

tickers = ['MSFT']


for ticker in tickers:
    data = yf.download(ticker, period='2y',
                       group_by='ticker', interval='1d')
    ticker_data = data['Close']

    # Split data to cut last known 30 days and make a prediction for these days
    # to compare prediction and real data for the last period:
    ticker_data = ticker_data[:-working_days]

    ma10 = ticker_data.rolling(window=10).mean()
    ma50 = ticker_data.rolling(window=50).mean()

    ema10 = ticker_data.ewm(span=10, adjust=False).mean()
    ema50 = ticker_data.ewm(span=50, adjust=False).mean()

    ticker_data.reset_index()

    df = ticker_data.reset_index()
    df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

    # Normalize numbers:
    df['y'] = df['y'] / df['y'].abs().max()
    ma10 = ma10 / ma10.abs().max()
    ma50 = ma50 / ma50.abs().max()
    ema10 = ema10 / ema10.abs().max()
    ema50 = ema50 / ema50.abs().max()

    """
    graph = go.Figure()
    graph.add_scatter(x=df['ds'], y=df['y'],
                      name=f'{ticker} Closed price')
    graph.add_scatter(x=df['ds'], y=ma10, name='MA 10', mode='lines')
    graph.add_scatter(x=df['ds'], y=ma50, name='MA 50', mode='lines')
    graph.add_scatter(x=df['ds'], y=ema10, name='EMA 10', mode='lines')
    graph.add_scatter(x=df['ds'], y=ema50, name='EMA 50', mode='lines')
    
    
    graph.update_layout(height=1000, width=1500)
    graph.show()
    """
