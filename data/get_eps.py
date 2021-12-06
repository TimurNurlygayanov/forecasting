import random
from finvizfinance.quote import finvizfinance


with open('data/revolut_tickers.txt', 'r', encoding='utf-8') as tickers_file:
    TICKERS = tickers_file.readlines()

TICKERS = [t.strip() for t in TICKERS if t]
random.shuffle(TICKERS)
TICKERS = TICKERS[:10]

for ticker in TICKERS:
    stock = finvizfinance(ticker)

    data = stock.TickerFundament()

    if data['EPS (ttm)'] > '-':
        parameter = 100 * float(data['EPS (ttm)']) / float(data['Price'])
        volume = int(data['Volume'].replace(',', ''))
        dept = float(data["Debt/Eq"].replace('-', '0'))

        if dept < 2 and volume > 200000:
            print(f'{ticker} {parameter:.1f}%/year {volume}')
            print(f'Debt/Eq: {dept}, ROE: {data["ROE"]}, ROI: {data["ROI"]}')
            print('- ' * 10)
    # print(data)
    # exit(1)
