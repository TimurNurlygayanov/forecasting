import requests
import pandas as pd


def get_data_alpha(ticker, interval='15min', limit=100):  # Daily
    with open('/Users/timur.nurlygaianov/alpha_key.txt', mode='r', encoding='utf8') as f:
        api_token = f.read().strip()

    ticker1 = ticker[:3]
    ticker2 = ticker[3:]

    data_type = 'FX_INTRADAY'
    if interval == 'Daily':
        data_type = 'FX_DAILY'

    outputsize = 'full'
    if limit <= 100:
        outputsize = 'compact'

    url = (f'https://www.alphavantage.co/query?function={data_type}&'
           f'from_symbol={ticker1}&to_symbol={ticker2}&interval={interval}&apikey={api_token}&outputsize={outputsize}')
    r = requests.get(url)
    data = r.json()

    indexes = []
    pd_data = {'Close': [], 'Open': [], 'Low': [], 'High': []}

    for date, values in list(reversed(data[f'Time Series FX ({interval})'].items()))[-limit:]:
        if len(indexes) < limit:
            indexes.append(date)

            pd_data['Open'].append(float(values['1. open']))
            pd_data['Close'].append(float(values['4. close']))
            pd_data['High'].append(float(values['2. high']))
            pd_data['Low'].append(float(values['3. low']))

    df = pd.DataFrame(pd_data, index=indexes)
    df.sort_index()

    return df
