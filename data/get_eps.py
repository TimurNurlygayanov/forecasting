import random
from finvizfinance.quote import finvizfinance


with open('data/revolut_tickers.txt', 'r', encoding='utf-8') as tickers_file:
    TICKERS = tickers_file.readlines()


TICKERS = [t.strip() for t in TICKERS if t]
random.shuffle(TICKERS)
TICKERS = TICKERS

TOP_TABLE = []

for ticker in TICKERS:
    print(f'Reading data for {ticker}... ')

    try:
        stock = finvizfinance(ticker)
        data = stock.TickerFundament()
    except:
        continue

    if data['EPS (ttm)'] > '-':
        price = float(data['Price'])
        earning_per_dollar = 100 * float(data['EPS (ttm)']) / price
        volume = int(data['Volume'].replace(',', ''))
        dept = float(data['Debt/Eq'].replace('-', '0'))
        roe = float(data['ROE'].replace('-', '0').replace('%', ''))
        roi = float(data['ROI'].replace('-', '0').replace('%', ''))
        target_price = float(data['Target Price'].replace('-', '0'))

        past_eps = -1
        if data['EPS past 5Y'] != '-':
            past_eps = float(data['EPS past 5Y'].replace('%', ''))

        future_eps = -1
        if data['EPS next 5Y'] != '-':
            future_eps = float(data['EPS next 5Y'].replace('%', ''))

        insiders_own = 0
        if data['Insider Own'] != '-':
            insiders_own = float(data['Insider Own'].replace('%', ''))

        if dept < 5 and earning_per_dollar > 2:  #  and volume > 300000:
            if roe > 0 and roi > 0 and past_eps > 0 and future_eps > 0:
                if price * 1.2 < target_price:
                    print(ticker, earning_per_dollar, past_eps)

                    TOP_TABLE.append({
                        'ticker': ticker,
                        '$_per_dollar': round(earning_per_dollar, 1),
                        'roe': roe,
                        'roi': roi,
                        'price': round(price, 2),
                        'past_eps': past_eps,
                        'insiders': insiders_own
                    })

TOP_TABLE = sorted(TOP_TABLE,
                   key=lambda x: x['$_per_dollar'],
                   reverse=True)
for i, res in enumerate(TOP_TABLE):
    TOP_TABLE[i]['top1'] = i

TOP_TABLE = sorted(TOP_TABLE,
                   key=lambda x: x['roi'],
                   reverse=True)
for i, res in enumerate(TOP_TABLE):
    TOP_TABLE[i]['top2'] = i

TOP_TABLE = sorted(TOP_TABLE,
                   key=lambda x: x['top1'] + x['top2'])

for res in TOP_TABLE:
    print(res)
