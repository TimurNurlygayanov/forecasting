import datetime

import pandas as pd
import pandas_ta  # for TA magic


def search(ticker='AAPL', period='700d'):
    df = pd.DataFrame()
    df = df.ta.ticker(ticker, period=period, interval="1h")

    df.ta.supertrend(append=True, length=10, multiplier=3.0,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s', ))
    df.ta.rsi(length=14, append=True, col_names=('RSI',))
    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    if df['S_trend_d'][-1] > 0:
        if df['RSI'][-1] < 50:
            if df['MACD'][-1] > df['MACD_signal'][-1]:
                buy_price = df['Close'][-1]
                stop_loss = df['S_trend_l'][-1] * 0.99
                stop_loss_percentage = 100 * (buy_price - stop_loss) / buy_price
                take_profit = buy_price + 2.1 * (buy_price - stop_loss)

                # print(ticker)
                # print(f"Buy price: {buy_price}, stop loss: {stop_loss} ({stop_loss_percentage:.1f}%), take profit: {take_profit}")
                # print('------')

                print(f"{ticker} {buy_price:.2f} {stop_loss:.2f} {take_profit:.2f}".replace('.', ','))


if __name__ == '__main__':
    print('=' * 20)
    print(datetime.datetime.now().strftime("%d %B %Y, %H:%M"))
    print('=' * 20)

    with open('smp500.txt', 'r') as f:
        TICKERS = f.readlines()

    TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
    TICKERS.remove('CEG')
    TICKERS.remove('ELV')
    TICKERS.remove('TWTR')
    TICKERS.remove('NLOK')
    TICKERS.remove('FBHS')

    for ticker in TICKERS:
        search(ticker)
