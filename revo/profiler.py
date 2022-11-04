import pandas as pd
import numpy as np
import pandas_ta  # for TA magic
import vectorbt as vbt


class Profiler:

    data = {}
    RSI_PERIOD = 14
    RSI_THRESHOLD = 33
    BBANDS_PERIOD = 34
    BBANDS_STD = 2.3
    TAKE_PROFIT_THRESHOLD = 1.07  # if price is higher than 10% from last buy, sell it
    MAX_LENGTH_FOR_BET = 50
    budget = 10000   # initial budget
    deal_lower_threshold = 300  # do not trade less than $1000
    positions = {}
    period = 250

    def __init__(self, tickers: list, period: int):
        self.period = period

        for i, ticker in enumerate(tickers):
            pre_profile_period = 250

            df = pd.DataFrame()
            df = df.ta.ticker(ticker, period=f'{period + pre_profile_period}d')

            if df is None or len(df) < period:  # remove all tickers that don't have required data
                continue

            df.ta.rsi(length=self.RSI_PERIOD, append=True, col_names=('RSI',))
            df.ta.bbands(length=self.BBANDS_PERIOD, std=self.BBANDS_STD, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

            # df.ta.sma(length=100, append=True, col_names=('SMA_100',))
            # df.ta.ema(length=7, append=True, col_names=('EMA_7',))
            # df.ta.ema(length=21, append=True, col_names=('EMA_21',))

            # df.ta.adx(append=True, col_names=('ADX', 'DMP', 'DMN'))

            self.data[ticker] = self.apply_strategy(df)

            pf = vbt.Portfolio.from_signals(df['Close'].values[:-period],
                                            entries=df['buy_signals'].values[:-period],
                                            exits=df['sell_signals'].values[:-period],
                                            freq='D',
                                            init_cash=10_000, fees=0.0025, slippage=0.0025)
            results = pf.stats()

            print(f'{ticker} {results["Total Return [%]"]:.1f}%' + ' ' * 10)
            if results['Total Return [%]'] > 10:
                self.data[ticker] = df[-period:]
            else:
                del self.data[ticker]

            print(f'Loading {100 * i/len(tickers):.1f}%...', end='\r')

    def buy(self, ticker, current_price):
        if ticker not in self.positions:
            buy_price = min(self.budget, 3000)

            if buy_price > self.deal_lower_threshold:
                shares_count = float(buy_price / current_price)
                self.budget -= shares_count * current_price
                self.positions[ticker] = shares_count

    def sell(self, ticker, current_price):
        if ticker in self.positions:
            self.budget += current_price * self.positions[ticker]

            del self.positions[ticker]

    def apply_strategy(self, df):
        buy_signals = {}
        sell_signals = {}
        last_buy_position = 0

        for i, (index, row) in enumerate(df.iterrows()):
            buy_signals[i] = False
            sell_signals[i] = False

            previous_p = df['P'].values[i - 1]

            if row['P'] > 0 > previous_p:
                buy_signals[i] = True
                last_buy_position = i

            if row['RSI'] < self.RSI_THRESHOLD:
                buy_signals[i] = True
                last_buy_position = i

            # RSI is too high, sell:
            if row['RSI'] > 70:
                sell_signals[i] = True

            if row['P'] > 80:
                sell_signals[i] = True

            # Sell as soon as we got desired profit:
            if row['Close'] > self.TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                sell_signals[i] = True

            if i - last_buy_position > self.MAX_LENGTH_FOR_BET:
                if row['Close'] > 1.02 * df['Close'].values[last_buy_position]:
                    sell_signals[i] = True

        df['buy_signals'] = buy_signals.values()
        df['sell_signals'] = sell_signals.values()

        return df

    def backtest(self):
        # skip first 50 days since we don't have enough data for this period
        for i in range(50, self.period):
            for ticker in self.data:
                if self.data[ticker]['buy_signals'].values[i] == True:
                    if self.budget > self.deal_lower_threshold:
                        self.buy(ticker, current_price=self.data[ticker]['Close'].values[i])

                if self.data[ticker]['sell_signals'].values[i] == True:
                    self.sell(ticker, current_price=self.data[ticker]['Close'].values[i])

            total_equity = self.budget
            for ticker in self.positions:
                total_equity += self.positions[ticker] * self.data[ticker]['Close'].values[i]

            stats = (f'day {i}, current budget: ${self.budget:.0f}, total ${total_equity:.0f} '
                     f' profit: {100.0 * (total_equity / 10000) - 100:.2f} %'
                     f' positions: {self.positions}')
            print(stats)


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]

p = Profiler(tickers=TICKERS[:10], period=250)
p.backtest()

