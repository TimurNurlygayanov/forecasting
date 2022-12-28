import numpy as np
import gym
from gym import spaces

import pandas as pd
import pandas_ta  # for TA magic

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from os import path

from plotly.subplots import make_subplots


# Simplify rewards
# Train with one share to win most of the time first
# Validate & improve input data

INITIAL_DEPOSIT = 3000    # The agent starts with $3000 deposit
ORDER_FEES = 1            # $1 for execution of each order
STATE_DATA = 5  # 7*5


class ProfitEnv(gym.Env):
    df = None
    current_index = 0
    last_bought_position = 0
    last_bought_price = 0
    budget = 0
    number_of_shares_bought = 0
    total_deals = 0
    metadata = {'render.modes': ['human']}
    ticker = ''
    mode = ''
    sell_signals = None
    buy_signals = None
    sum_profit = 0
    tickers = None
    initial_tickers = []

    _max_episode_steps = 5000  # get profit in 10 days

    def __init__(self, tickers_list: list = None, mode='learn'):
        super(ProfitEnv, self).__init__()

        self.mode = mode

        self.action_space = spaces.Discrete(3)

        # Current price: Low, High, Open, Close
        # EMA3, EMA7, EMA20, EMA70, EMA200, SMA50, SMA200, SMA500, SMA2000,
        # Bollinger_low, Bollinger_up, RSI, SuperTrend,
        # MACD, MACD_signal, ATR, current profit,
        self.observation_space = spaces.Box(low=-10, high=10, shape=(STATE_DATA, 28), dtype=np.float32)

        self.tickers = tickers_list.copy()
        self.initial_tickers = tickers_list.copy()

        self.ticker = self.tickers.pop()
        self.get_data(self.ticker)

        self.reset()

    def prepare_data(self):
        try:
            self.df.ta.ema(length=3, append=True, col_names=('EMA3',))
            self.df.ta.ema(length=7, append=True, col_names=('EMA7',))
            self.df.ta.ema(length=20, append=True, col_names=('EMA20',))
            self.df.ta.ema(length=35, append=True, col_names=('EMA35',))

            self.df.ta.ema(length=7*8, append=True, col_names=('EMA70',))
            self.df.ta.ema(length=20*8, append=True, col_names=('EMA200',))

            self.df.ta.sma(length=50, append=True, col_names=('SMA50',))
            self.df.ta.sma(length=200, append=True, col_names=('SMA200',))

            self.df.ta.sma(length=50*8, append=True, col_names=('SMA500',))
            self.df.ta.sma(length=200*8, append=True, col_names=('SMA2000',))

            self.df.ta.rsi(append=True, col_names=('RSI',))
            self.df['RSI'] /= 100

            self.df.ta.rsi(length=8*14, append=True, col_names=('RSI_LONG',))
            self.df['RSI_LONG'] /= 100

            self.df.ta.atr(append=True, col_names=('ATR',))
            self.df['ATR'] /= 100

            self.df.ta.bbands(length=34, col_names=('L', 'M', 'U', 'B', 'P'), append=True)

            self.df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

            self.df.ta.supertrend(append=True, length=10, multiplier=4.0,
                                  col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
        except Exception as e:
            print(f'WRONG DATA {self.ticker} {e}')
            exit(1)

        self.df = self.df[200 * 8:].copy()

    def get_state(self):
        states = []

        for i in range(self.current_index - STATE_DATA, self.current_index):
            state = []

            scaled_parameters = [
                'Low', 'High', 'Open', 'Close', 'EMA3', 'EMA7', 'EMA20', 'EMA70', 'EMA200',
                'SMA50', 'SMA200', 'SMA500', 'SMA2000', 'L', 'U'
            ]

            for parameter in scaled_parameters:
                state.append(self.df[parameter].values[i] / self.df['SMA200'].values[self.current_index] - 1)

            """
            if i == 0:
                state.append(1)
            else:
                state.append(self.df['Volume'].values[i] / max(self.df['Volume'].values[:i]))
            """

            parameters = ['RSI', 'RSI_LONG', 'S_trend_d', 'MACD', 'MACD_signal', 'ATR']
            for parameter in parameters:
                state.append(self.df[parameter].values[i])

            # Calculate current profit:
            profit = 0
            if self.last_bought_price > 0:
                profit = self.df['Close'].values[i] / self.last_bought_price - 1
            state.append(profit)

            # Check flags:
            ema20_flag = -1
            if self.df['EMA20'].values[i] > self.df['SMA50'].values[i]:
                ema20_flag = 1

            sma50_flag = -1
            if self.df['SMA50'].values[i] > self.df['SMA200'].values[i]:
                sma50_flag = 1

            sma500_flag = -1
            if self.df['SMA500'].values[i] > self.df['SMA2000'].values[i]:
                sma500_flag = 1

            state.append(ema20_flag)
            state.append(sma50_flag)
            state.append(sma500_flag)

            # Our own moving average crossover for daily trades
            diff = self.df['EMA7'].values[i] / self.df['EMA35'].values[i] - 1
            state.append(diff)

            diff = self.df['EMA35'].values[i] / self.df['SMA200'].values[i] - 1
            state.append(diff)

            # Check Volume diff
            volume_increases = 0
            if i > 0 and self.df['Volume'].values[i] > self.df['Volume'].values[i-1]:
                volume_increases = 1
            state.append(volume_increases)

            states.append(state)

        return np.array(states, dtype=np.float32)

    def step(self, action):
        # actions: 0 - hold, 1 - buy, 2 - sell

        reward = 0

        """
        if action == 0:
            # Punish for not buying shares:
            if self.last_bought_position == 0:
                reward = -0.02  # deduct 2 % for each date when we didn't have any shares

            # Punish for holding without the profit:
            current_price = self.df['Close'].values[self.current_index]

            if 0 < current_price < self.last_bought_price:
                reward = -1
            elif self.last_bought_position > 0:
                if current_price < 0.9 * max(self.df['Close'].values[self.last_bought_position:self.current_index]):
                    reward = -0.01

            if 0 < self.last_bought_price < current_price:  # Reward for holding with profit:
                reward = 0.1

                if current_price > 0.95 * max(self.df['Close'].values[self.last_bought_position:self.current_index]):
                    reward = 0.2
        """

        if action == 0:
            if self.last_bought_price > 0:
                current_price = self.df['Close'].values[self.current_index]
                reward = current_price / self.last_bought_price - 1

        if action == 1:
            if self.last_bought_position == 0:
                current_price = self.df['Close'].values[self.current_index]

                bet_size = min(self.budget, 20000)  # limit size of every deal

                number_of_shares = 10  # bet_size // current_price

                if number_of_shares > 0:
                    self.budget -= number_of_shares * current_price

                    self.last_bought_price = current_price
                    self.last_bought_position = self.current_index
                    self.number_of_shares_bought = number_of_shares
                    self.total_deals += 1

                    self.buy_signals[self.current_index] = True

                    reward = -2  # reward for making a bet - make it profitable only if we got +2% from bet

        if action == 2:
            if self.last_bought_position > 0:  # sell only if we have something to sell
                current_price = self.df['Close'].values[self.current_index]

                gross_profit = current_price * self.number_of_shares_bought

                profit_percent = current_price / self.last_bought_price - 1
                if profit_percent > 0:
                    net_profit = gross_profit - 0.26 * profit_percent * gross_profit  # make sure we discount the taxes
                else:
                    net_profit = gross_profit

                self.budget += net_profit
                self.budget -= 2 * ORDER_FEES  # deduct broker fees

                with open(f"rl/deals_{self.mode}.txt", 'a+') as myfile:
                    myfile.write(f'sell {self.ticker} profit: '
                                 f'{100 * (current_price / self.last_bought_price - 1):.1f}% '
                                 f'total: ${self.budget:.0f}\n')

                # positive reward in case we got profit, negative reward otherwise
                profit_percent = current_price / self.last_bought_price - 1
                reward = int(profit_percent * 100)

                # if we got large profit or loss, multiply the reward/punishment to highlight the impact
                if abs(reward) > 10:
                    reward = int(reward * 1.2)

                """
                if reward < 0:
                    reward = int(reward * 2)
                """

                self.sum_profit += reward

                self.sell_signals[self.current_index] = True

                self.number_of_shares_bought = 0
                self.last_bought_price = 0
                self.last_bought_position = 0

        self.current_index += 1
        done = bool(self.current_index >= len(self.df['Close']) - 1)

        if done and len(self.tickers) > 0:
            done = False

            ticker = self.tickers.pop()
            # print(f'new ticker {ticker} left: {len(self.tickers)}')

            self.get_data(ticker)
            self.prepare_data()
            self.reset(cleanup_budget=False)
        elif not self.tickers and self.mode != 'test':
            self.tickers = self.initial_tickers.copy()

        budget = self.budget + self.last_bought_price * self.number_of_shares_bought
        if budget < 500:
            self.budget = 3000

        return self.get_state(), reward, done, {}

    def get_data(self, ticker):
        file_name = f'rl/data/daily_{ticker}.xlsx'
        self.ticker = ticker

        if path.exists(file_name):
            self.df = pd.read_excel(file_name, index_col=0)  # , header=[0, 1]
        else:
            self.df = pd.DataFrame()
            self.df = self.df.ta.ticker(ticker, period='700d', interval='1h')
            self.prepare_data()

            self.df.index = self.df.index.strftime('%m/%d/%Y %H:%M')
            self.df.to_excel(file_name, index=True, header=True)

    def reset(self, cleanup_budget=True):
        if cleanup_budget:
            self.budget = INITIAL_DEPOSIT
        else:
            if self.last_bought_position > 0:
                self.budget += self.last_bought_price * self.number_of_shares_bought

        self.current_index = STATE_DATA  # we start from the Day #5 in the data set to have a history
        self.last_bought_position = 0
        self.last_bought_price = 0
        self.number_of_shares_bought = 0
        self.sum_profit = 0

        self.buy_signals = {i: False for i in range(len(self.df))}
        self.sell_signals = {i: False for i in range(len(self.df))}

        state = self.get_state()
        # print(state)
        return state

    def draw(self):
        df = self.df

        graph = make_subplots(rows=1, cols=1)
        graph.update_layout(title=self.ticker, xaxis_rangeslider_visible=False)

        graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

        graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                          line={'color': 'magenta'})
        graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
                          line={'color': 'blue', 'width': 3})

        buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
        graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                          marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

        sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
        graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                          marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

        graph.show()

    def render(self, mode='human', close=False):
        if self.total_deals > 0 and self.total_deals % 100 == 0:
            total_budget = self.budget
            if self.last_bought_position > 0:
                total_budget += self.last_bought_price * self.number_of_shares_bought

            message = f"""
            Current budget: ${total_budget:.1f}
            Profit: {100 * (total_budget / INITIAL_DEPOSIT - 1):.0f}%
            Total Deals: {self.total_deals}
            """
            print(message)

        if self.current_index >= len(self.df) - 2:
            self.df['buy_signals'] = self.buy_signals.values()
            self.df['sell_signals'] = self.sell_signals.values()
            self.draw()

        """
        if self.mode == 'test':
            print(self.current_index)
            print(self.get_state())
        """


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
TICKERS.remove('CEG')
TICKERS.remove('ELV')

non_vectorized_env = ProfitEnv(TICKERS[:1].copy())
env = DummyVecEnv([lambda: non_vectorized_env])

model = PPO('MlpPolicy', env, verbose=True, tensorboard_log='logs')  # learning_rate=0.0001
model.learn(total_timesteps=500000)

print(TICKERS[:5])
non_vectorized_env = ProfitEnv(TICKERS[:1].copy(), mode='test')
env = DummyVecEnv([lambda: non_vectorized_env])

evaluate_policy(model, env, n_eval_episodes=1, render=True)
