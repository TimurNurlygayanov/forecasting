#
# python3 rl/rl_daily/env_buy_only.py
# tensorboard --logdir logs
#


import numpy as np
import gym
from gym import spaces

import pandas as pd
import pandas_ta  # for TA magic

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise
from gym.envs.registration import EnvSpec
from stable_baselines3.common.env_checker import check_env

from os import path

from plotly.subplots import make_subplots


INITIAL_DEPOSIT = 3000    # The agent starts with $3000 deposit
ORDER_FEES = 1            # $1 for execution of each order
STATE_DATA = 1
PROFIT_PERIOD = 200


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
    spec = EnvSpec(id='ProfitEnv-v0', max_episode_steps=10000, reward_threshold=5000)

    _max_episode_steps = PROFIT_PERIOD * 10  # get profit in 50 days

    def __init__(self, tickers_list: list = None, mode='learn'):
        super(ProfitEnv, self).__init__()

        self.mode = mode

        self.action_space = spaces.Discrete(2)  # hold or buy only

        # Current price: Low, High, Open, Close
        # EMA3, EMA7, EMA20, EMA70, EMA200, SMA50, SMA200, SMA500, SMA2000,
        # Bollinger_low, Bollinger_up, RSI, SuperTrend,
        # MACD, MACD_signal, ATR, current profit,
        self.observation_space = spaces.Box(low=-1, high=1, shape=(STATE_DATA, 25), dtype=np.float16)

        self.tickers = tickers_list.copy()
        self.initial_tickers = tickers_list.copy()

        self.ticker = self.tickers.pop()
        self.get_data(self.ticker)

        self.reset()

    def prepare_data(self):
        k_period = 100
        k_indicator = {i: 0 for i in range(200)}
        k2_indicator = {i: 0 for i in range(200)}

        try:
            self.df.ta.ema(length=3, append=True, col_names=('EMA3',))
            self.df.ta.ema(length=7, append=True, col_names=('EMA7',))
            self.df.ta.ema(length=20, append=True, col_names=('EMA20',))

            self.df.ta.ema(length=7 * 8, append=True, col_names=('EMA70',))
            self.df.ta.ema(length=14 * 8, append=True, col_names=('EMA140',))
            self.df.ta.ema(length=21 * 8, append=True, col_names=('EMA200',))

            self.df.ta.sma(length=50, append=True, col_names=('SMA50',))
            self.df.ta.sma(length=200, append=True, col_names=('SMA200',))

            self.df.ta.sma(length=50*8, append=True, col_names=('SMA500',))
            self.df.ta.sma(length=200*8, append=True, col_names=('SMA2000',))

            self.df.ta.rsi(append=True, col_names=('RSI',))
            self.df['RSI'] /= 100

            self.df.ta.atr(append=True, col_names=('ATR',))
            self.df['ATR'] /= 100

            self.df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)   # length=34,

            self.df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

            self.df.ta.supertrend(append=True, length=8*10, multiplier=4.0,
                                  col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

            # Calculate our own metrics:

            self.df['K2'] = (self.df['High'] - self.df['Low']) / self.df['L']
            self.df['K'] = (self.df['EMA20'] - self.df['EMA7']) / self.df['Close'] - 0.01

            self.df['diff1'] = self.df['EMA7'] / self.df['SMA50'] - 1
            self.df['diff2'] = (self.df['EMA7'] - self.df['EMA3']) / self.df['L']

            for i in range(200, len(self.df['Close'])):
                # Integral-based metrics
                diff = self.df['SMA200'].values[i - k_period:i] - self.df['EMA7'].values[i - k_period:i]

                prob_up = diff[diff > 0].sum()
                prob_down = abs(diff[diff < 0].sum())

                k_indicator[i] = prob_up / (prob_up + prob_down)

                diff = self.df['EMA7'].values[i - k_period:i] - self.df['EMA3'].values[i - k_period:i]
                prob_up = diff[diff > 0].sum()
                prob_down = abs(diff[diff < 0].sum())

                k2_indicator[i] = prob_up / (prob_up + prob_down)

            self.df['W'] = k_indicator.values()
            self.df['W2'] = k2_indicator.values()
        except Exception as e:
            print(f'WRONG DATA {self.ticker} {e}')
            exit(1)

        self.df = self.df[200 * 8:].copy().fillna(0)

    def get_state(self):
        states = []

        for i in range(self.current_index - STATE_DATA, self.current_index):
            state = []

            scaled_parameters = [
                'Low', 'High', 'Open', 'Close', 'EMA3', 'EMA7', 'EMA20', 'EMA70', 'EMA140', 'EMA200',
                'SMA50', 'SMA200', 'SMA500', 'SMA2000', 'L', 'U',
            ]

            for parameter in scaled_parameters:
                state.append(self.df[parameter].values[i])

            parameters = ['RSI', 'MACD', 'S_trend_d', 'K', 'K2']   # 'S_trend_d', 'MACD', 'MACD_signal', 'ATR',
            for parameter in parameters:
                state.append(self.df[parameter].values[i])

            # Our own moving average crossover for daily trades
            state.append(self.df['diff1'].values[i])
            state.append(self.df['diff2'].values[i])

            state.append(self.df['W'].values[i])
            state.append(self.df['W2'].values[i])

            state = np.array(state, dtype=np.float16).flatten()
            states.append(state)

        return states

    def step(self, action):
        # actions: 0 - hold, 1 - buy, 2 - sell

        reward = 0

        current_price = self.df['Close'].values[self.current_index]
        start = self.current_index + 1
        end = start + PROFIT_PERIOD

        max_loss = 0.05
        profit_to_loss_rate = 2

        stop_loss = current_price * (1 - max_loss)
        take_profit = current_price * (1 + max_loss * profit_to_loss_rate)

        profit = 0

        for i in range(start, end):
            if self.df['Low'].values[i] <= stop_loss:
                profit = (stop_loss - current_price) / current_price
            if self.df['High'].values[i] >= take_profit:
                profit = (take_profit - current_price) / current_price

        if action == 1:
            # bet_size = min(self.budget, 20000)  # limit size of every deal

            number_of_shares = 10  #  bet_size // current_price

            self.last_bought_price = current_price
            self.last_bought_position = self.current_index
            self.number_of_shares_bought = number_of_shares
            self.total_deals += 1

            self.buy_signals[self.current_index] = True

            reward = profit

            with open(f"rl/deals_{self.mode}.txt", 'a+') as myfile:
                myfile.write(f'sell {self.ticker} profit: '
                             f'{100 * profit:.1f}% reward: {reward:.1f}\n')

        self.current_index += 1
        done = bool(self.current_index >= len(self.df['Close']) - PROFIT_PERIOD)

        if done and len(self.tickers) > 0:
            done = False

            ticker = self.tickers.pop()

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

        scaled_parameters = [
            'Low', 'High', 'Open', 'Close', 'EMA3', 'EMA7', 'EMA20', 'EMA70', 'EMA140', 'EMA200',
            'SMA50', 'SMA200', 'SMA500', 'L', 'U'
        ]

        for parameter in scaled_parameters:
            self.df[parameter] = self.df[parameter] / self.df['SMA2000'] - 1

        self.df['SMA2000'] = 1

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

        if self.current_index >= len(self.df) - PROFIT_PERIOD-1:
            self.df['buy_signals'] = self.buy_signals.values()
            self.df['sell_signals'] = self.sell_signals.values()
            self.draw()


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]
TICKERS.remove('CEG')
TICKERS.remove('ELV')

non_vectorized_env = ProfitEnv(TICKERS[:100].copy())
# wrapped_env = gym.wrappers.Monitor(non_vectorized_env, './results', force=True)
env = DummyVecEnv([lambda: non_vectorized_env])

state = env.reset()
print(state)

model = PPO('MlpPolicy', env, verbose=True, tensorboard_log='logs')  # learning_rate=0.0001
model.learn(total_timesteps=500000)


non_vectorized_env = ProfitEnv(TICKERS[:1].copy(), mode='test')
env = gym.wrappers.Monitor(non_vectorized_env, './results', force=True)
# env = DummyVecEnv([lambda: wrapped_env])

evaluate_policy(model, env, n_eval_episodes=1, render=True)
