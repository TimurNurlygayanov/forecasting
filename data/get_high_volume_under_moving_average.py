import random
from os import path
import pandas as pd
from finvizfinance.quote import finvizfinance

prediction_period = 60
working_days = 5 * prediction_period // 7

with open('data/revolut_tickers.txt', 'r', encoding='utf-8') as tickers_file:
    TICKERS = tickers_file.readlines()


TICKERS = [t.strip() for t in TICKERS if t]
random.shuffle(TICKERS)
TICKERS = TICKERS

TOP_TABLE = []

data_file_name = 'data/tickers/historical.xlsx'
if path.exists(data_file_name):
    historical_data = pd.read_excel(data_file_name, index_col=0, header=[0, 1])

for ticker in TICKERS[:20]:
    print(f'Reading data for {ticker}... ')

    try:
        stock = finvizfinance(ticker)
        data = stock.ticker_fundament()
    except Exception as e:
        print(e)
        continue

    if ticker not in historical_data:
        continue

    ticker_data = historical_data[ticker]['Close'][:-working_days]
    ema50 = ticker_data.ewm(span=50, adjust=False).mean()

    volume = int(data['Volume'].replace(',', ''))

    print(volume)
    print(ema50)

    if data['EPS (ttm)'] > '-':
        price = float(data['Price'])
        earning_per_dollar = 100 * float(data['EPS (ttm)']) / price
        volume = int(data['Volume'].replace(',', ''))
        dept = float(data['Debt/Eq'].replace('-', '0'))
        roe = float(data['ROE'].replace('-', '0').replace('%', ''))
        roi = float(data['ROI'].replace('-', '0').replace('%', ''))
        target_price = float(data['Target Price'].replace('-', '0'))
        peg = float(data['PEG'].replace('-', '0'))
        price_to_book = float(data['P/B'].replace('-', '0'))

        past_eps = -1
        if data['EPS past 5Y'] != '-':
            past_eps = float(data['EPS past 5Y'].replace('%', ''))

        future_eps = -1
        if data['EPS next 5Y'] != '-':
            future_eps = float(data['EPS next 5Y'].replace('%', ''))

        insiders_own = 0
        if data['Insider Own'] != '-':
            insiders_own = float(data['Insider Own'].replace('%', ''))

        print(volume)
        print(ema50)

        if volume > 300000:
            if ema50[-1] > price:
                print(ticker)
