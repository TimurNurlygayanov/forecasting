import pandas as pd
import numpy as np
import pandas_ta  # for TA magic
import vectorbt as vbt
from plotly.subplots import make_subplots


def draw(ticker, df):
    graph = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.01,)
    graph.update_layout(title=ticker)

    graph.add_scatter(y=df['Close'], mode='lines', name='Price',
                      line={'color': 'green', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                      line={'color': 'magenta'}, row=1, col=1)
    graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                      line={'color': 'blue', 'width': 3}, row=1, col=1)

    graph.add_scatter(y=df['S_trend'], mode='lines', name='S_trend', row=1, col=1)
    graph.add_scatter(y=df['S_trend_s'], mode='lines', name='S_trend_s',
                      line={'color': '#ff4040', 'width': 3}, row=1, col=1)
    graph.add_scatter(y=df['S_trend_l'], mode='lines', name='S_trend_l SUPER TREND',
                      line={'color': '#00ff7f', 'width': 3}, row=1, col=1)

    buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
    graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                      marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

    sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
    graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                      marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

    graph.show()


class Profiler:

    data = {}
    TAKE_PROFIT_THRESHOLD = 1.50  # if price is higher than 50% from last buy, sell it
    STOP_LOSSES_THRESHOLD = 0.92  # stop losses at 8%
    initial_budget = 10000  # initial budget
    budget = 0
    positions = {}
    period = 250

    def __init__(self, tickers: list, period: int):
        self.period = period
        self.budget = self.initial_budget
        self.max_bet = self.initial_budget / 4

        for i, ticker in enumerate(tickers):
            df = pd.DataFrame()
            df = df.ta.ticker(ticker, period=f'{period+200}d')  # add extra 200 days here to calculate 200 SMA

            if df is None or len(df) < period+200:  # remove all tickers that don't have required data
                continue

            df.ta.supertrend(length=10, multiplier=4.0, append=True,
                             col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

            df.ta.sma(length=200, append=True, col_names=('SMA200',))
            df.ta.sma(length=50, append=True, col_names=('SMA50',))
            df = df[200:].copy()  # cut first 200 days

            self.data[ticker] = self.apply_strategy(df)

            print(f'Loading {100 * i/len(tickers):.1f}%...', end='\r')

    def buy(self, ticker, current_price, moment=0):
        if ticker not in self.positions:
            buy_price = min(self.budget, self.max_bet)

            shares_count = int(buy_price / current_price)
            self.budget -= shares_count * current_price
            self.positions[ticker] = {'shares_count': shares_count, 'price': current_price, 'moment': moment}

    def sell(self, ticker, current_price, moment=0):
        if ticker in self.positions:
            self.budget += current_price * self.positions[ticker]['shares_count']

            profit = 100 * (current_price / self.positions[ticker]['price'] - 1)
            print(f'sold {ticker}, profit: {profit:.1f}')

            del self.positions[ticker]

    def apply_strategy(self, df):
        buy_signals = {}
        sell_signals = {}
        last_buy_position = 0

        for i, (index, row) in enumerate(df.iterrows()):
            buy_signals[i] = False
            sell_signals[i] = False

            if row['Close'] > row['SMA50'] > row['SMA200'] > 0:
                if df['S_trend_d'].values[i - 2] < 1 < df['S_trend_d'].values[i - 1] + row['S_trend_d']:
                    buy_signals[i] = True
                    last_buy_position = i

            if i > last_buy_position > 0:
                """
                # Sell if super trend already finished
                if row['S_trend_d'] < 0:
                    sell_signals[i] = True
                    last_buy_position = 0
                """

                # Sell as soon as we got total desired profit:
                if row['Close'] > self.TAKE_PROFIT_THRESHOLD * df['Close'].values[last_buy_position]:
                    sell_signals[i] = True

                # Stop loses at 8% of loses:
                # if row['Close'] < self.STOP_LOSSES_THRESHOLD * df['Close'].values[last_buy_position]:
                #    sell_signals[i] = True

                # Sell if we got more than 2 % daily rate
                current_profit = 1 - row['Close'] / df['Close'].values[last_buy_position]
                if current_profit / (i - last_buy_position) > 0.5:
                    sell_signals[i] = True

        df['buy_signals'] = buy_signals.values()
        df['sell_signals'] = sell_signals.values()

        return df

    def backtest(self):
        for i in range(self.period):
            for ticker in self.data:
                if self.data[ticker]['buy_signals'].values[i] == True:
                    if self.budget > self.max_bet:
                        self.buy(ticker, current_price=self.data[ticker]['Close'].values[i], moment=i)
                    else:
                        """
                        # find ticker that we hold for too long already:
                        ticker_to_sell = min(self.positions, key=lambda x: self.positions[x]['moment'])

                        # sell another ticker that has a most profitable conditions right now:
                        ticker_to_sell = max(self.positions,
                                             key=lambda x: self.data[x]['Close'].values[i] / self.positions[x]['price'])
                        """

                        ticker_to_sell = min(self.positions, key=lambda x: self.positions[x]['moment'])
                        # ticker_to_sell = max(self.positions,
                        #                      key=lambda x: self.data[x]['Close'].values[i] / self.positions[x]['price'])
                        profit = self.data[ticker_to_sell]['Close'].values[i] / self.positions[ticker_to_sell]['price']

                        if 100 * (profit - 1) < -5:  # sell only if the profit is less than -5%
                            print(f'profit calculated {profit}')

                            self.sell(ticker_to_sell, current_price=self.data[ticker_to_sell]['Close'].values[i], moment=i)

                            self.buy(ticker, current_price=self.data[ticker]['Close'].values[i], moment=i)

                if self.data[ticker]['sell_signals'].values[i] == True:
                    self.sell(ticker, current_price=self.data[ticker]['Close'].values[i], moment=i)

            total_equity = self.budget
            for ticker in self.positions:
                total_equity += self.positions[ticker]['shares_count'] * self.data[ticker]['Close'].values[i]

            stats = (f'day {i}, current budget: ${self.budget:.0f}, total ${total_equity:.0f} '
                     f' profit: {100.0 * (total_equity / self.initial_budget) - 100:.2f} %'
                     f' positions: {list(self.positions.keys())}')
            # print(stats)

        for ticker in self.positions:
            profit = 100 * (self.data[ticker]["Close"].values[-1] / self.positions[ticker]["price"] - 1)
            print(f'{ticker} profit: {profit:.1f}')

with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]

p = Profiler(tickers=TICKERS[:200], period=3 * 250)  # 1 year is 250 working days
p.backtest()
