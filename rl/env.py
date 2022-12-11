import numpy as np
import gym
from gym import spaces

import pandas as pd
import pandas_ta  # for TA magic

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from os import path


INITIAL_DEPOSIT = 3000    # The agent starts with $3000 deposit
ORDER_FEES = 1            # $1 for execution of each order
TICKERS = ['ALB', 'TSLA', 'HD', 'NEE', 'MSFT']
TICKERS_TO_VERIFY = ['A', 'AAPL', 'NVDA']


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

    _max_episode_steps = 100

    def __init__(self, tickers_list: list = None, mode='learn'):
        super(ProfitEnv, self).__init__()

        self.mode = mode

        self.action_space = spaces.Discrete(3)

        # Current price: Low, High, Open, Close
        # EMA3, EMA7, EMA20, SMA50, SMA200, Volume, RSI, SuperTrend, MACD, MACD_signal
        # current profit,
        self.observation_space = spaces.Box(low=-100, high=100, shape=(5, 15), dtype=np.float32)

        self.tickers = tickers_list.copy()

        self.ticker = self.tickers.pop()
        self.get_data(self.ticker)

        self.reset()

    def prepare_data(self):
        self.df.ta.ema(length=3, append=True, col_names=('EMA3',))
        self.df.ta.ema(length=7, append=True, col_names=('EMA7',))
        self.df.ta.ema(length=20, append=True, col_names=('EMA20',))

        self.df.ta.sma(length=50, append=True, col_names=('SMA50',))
        self.df.ta.sma(length=200, append=True, col_names=('SMA200',))

        self.df.ta.rsi(append=True, col_names=('RSI',))

        self.df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

        self.df.ta.supertrend(append=True, length=10, multiplier=4.0,
                              col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

        self.df = self.df[200:].copy()

    def get_state(self):
        states = []

        for i in range(self.current_index-5, self.current_index):
            state = []

            scaled_parameters = [
                'Low', 'High', 'Open', 'Close', 'EMA3', 'EMA7', 'EMA20', 'SMA50', 'SMA200',
            ]

            for parameter in scaled_parameters:
                state.append(self.df[parameter].values[i] / self.df['SMA200'].values[i])

            if i == 0:
                state.append(1)
            else:
                state.append(self.df['Volume'].values[i] / max(self.df['Volume'].values[:i]))

            parameters = ['RSI', 'S_trend_d', 'MACD', 'MACD_signal']
            for parameter in parameters:
                state.append(self.df[parameter].values[i])

            # Calculate current profit:
            profit = 0
            if self.last_bought_price > 0:
                profit = self.df['Close'].values[i] / self.last_bought_price

            state.append(profit)

            states.append(state)

        return np.array(states, dtype=np.float32)

    def step(self, action):
        # actions: 0 - hold, 1 - buy, 2 - sell

        reward = 0

        if self.last_bought_position == 0:
            reward = -0.005  # deduct 0.5 % for each date when we didn't have any shares

        if action == 1:
            if self.last_bought_position == 0:
                current_price = self.df['Close'].values[self.current_index]

                bet_size = min(self.budget, 10000)  # limit size of every deal

                number_of_shares = bet_size // current_price

                if number_of_shares > 0:
                    self.budget -= number_of_shares * current_price

                    self.last_bought_price = current_price
                    self.last_bought_position = self.current_index
                    self.number_of_shares_bought = number_of_shares
                    self.total_deals += 1

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
                self.budget -= 2  # -$2 for orders execution

                with open(f"rl/deals_{self.mode}.txt", 'a+') as myfile:
                    myfile.write(f'sell {self.ticker} profit: '
                                 f'{100 * (current_price / self.last_bought_price - 1):.1f}% '
                                 f'total: {self.budget:.0f}\n')

                # positive reward in case we got profit, negative reward otherwise
                reward = current_price / self.last_bought_price - 1

                self.number_of_shares_bought = 0
                self.last_bought_price = 0
                self.last_bought_position = 0

        if reward < 0:
            reward *= 5  # punish the agent for loosing money

        if reward > 1.05:
            reward *= 2  # reward agent for large profit

        self.current_index += 1
        done = bool(self.current_index >= len(self.df['Close']) - 1)

        if done and len(self.tickers) > 0:
            done = False

            ticker = self.tickers.pop()

            self.get_data(ticker)
            self.prepare_data()
            self.reset(cleanup_budget=False)

        return self.get_state(), reward, done, {}

    def get_data(self, ticker):
        file_name = f'rl/data/{ticker}.xlsx'
        self.ticker = ticker

        if path.exists(file_name):
            self.df = pd.read_excel(file_name, index_col=0)  # , header=[0, 1]
        else:
            self.df = pd.DataFrame()
            self.df = self.df.ta.ticker(ticker, period='1000d')
            self.prepare_data()

            self.df.index = self.df.index.strftime('%m/%d/%Y')
            self.df.to_excel(file_name, index=True, header=True)

    def reset(self, cleanup_budget=True):
        if cleanup_budget:
            self.budget = INITIAL_DEPOSIT
        else:
            if self.last_bought_position > 0:
                self.budget += self.last_bought_price * self.number_of_shares_bought

        self.current_index = 5  # we start from the Day #5 in the data set to have a history
        self.last_bought_position = 0
        self.last_bought_price = 0
        self.number_of_shares_bought = 0

        return self.get_state()

    def render(self, mode='human', close=False):
        if self.total_deals % 100 == 0:
            total_budget = self.budget
            if self.last_bought_position > 0:
                total_budget += self.last_bought_price * self.number_of_shares_bought

            message = f"""
            Current budget: ${total_budget:.1f}
            Profit: {100 * (total_budget / INITIAL_DEPOSIT - 1):.0f}%
            Total Deals: {self.total_deals}
            """
            print(message)


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]


non_vectorized_env = ProfitEnv(TICKERS[:100].copy())
env = DummyVecEnv([lambda: non_vectorized_env])

model = PPO('MlpPolicy', env, verbose=True, tensorboard_log='logs')  # learning_rate=0.0001
model.learn(total_timesteps=1000000)

non_vectorized_env = ProfitEnv(TICKERS[100:200].copy(), mode='test')
env = DummyVecEnv([lambda: non_vectorized_env])

evaluate_policy(model, env, n_eval_episodes=1, render=True)
