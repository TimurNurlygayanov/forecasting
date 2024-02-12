#
# python3 rl/rl_daily/env_buy_only.py
# tensorboard --logdir logs
#

import warnings
warnings.filterwarnings("ignore")

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

from bot.utils import get_data

from os import path

from plotly.subplots import make_subplots


INITIAL_DEPOSIT = 3000    # The agent starts with $3000 deposit
ORDER_FEES = 1            # $1 for execution of each order
STATE_DATA = 2
PROFIT_PERIOD = 20

ACTIONS = {0: 0, 1: 0, 2: 0}

class ProfitEnv(gym.Env):
    df = None
    current_index = 0
    last_bought_position = 0
    last_bought_price = 0
    stop_loss = 0
    take_profit = 0
    budget = 0
    number_of_shares_bought = 0
    total_deals = 0
    metadata = {'render.modes': ['human']}
    ticker = ''
    mode = ''
    sell_signals = None
    buy_signals = None
    hold_signals = None
    sum_profit = 0
    tickers = None
    initial_tickers = []
    failures = 0
    spec = EnvSpec(id='ProfitEnv-v0', max_episode_steps=2000, reward_threshold=5000)
    total_rewards = 0
    done = False

    _max_episode_steps = 2000  # PROFIT_PERIOD * 10  # get profit in 50 days

    def __init__(self, tickers_list: list = None, mode='learn'):
        super(ProfitEnv, self).__init__()

        self.mode = mode

        self.action_space = spaces.Discrete(2)  # hold or buy only

        # Current price: Low, High, Open, Close
        # EMA3, EMA7, EMA20, EMA70, EMA200, SMA50, SMA200,
        # Bollinger_low, Bollinger_up, RSI, SuperTrend,
        # MACD, MACD_signal, ATR, current profit,
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DATA, 142), dtype=np.float32)   # shape=(1, 20)

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

            self.df.ta.sma(length=50, append=True, col_names=('SMA50',))

            self.df.ta.rsi(append=True, col_names=('RSI',))

            self.df.ta.atr(append=True, col_names=('ATR',))

            self.df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)   # length=34,

            self.df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

            self.df.ta.supertrend(append=True, length=10, multiplier=3.0,
                                  col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

            for m in self.df.columns:
                if m != 'S_trend_d':
                    self.df[m] = np.log(self.df[m])
        except Exception as e:
            print(f'WRONG DATA {self.ticker} {e}')
            exit(1)

        self.df = self.df[50:].copy().fillna(0)

    def get_state(self):
        states = []

        for i in range(self.current_index - STATE_DATA, self.current_index):
            state = []

            scaled_parameters = [
                'Low', 'High', 'Open', 'Close', 'EMA3', 'EMA7', 'EMA20',
                'SMA50', 'L', 'U', 'RSI', 'MACD', 'MACD_signal', 'ATR'
            ]

            for parameter in scaled_parameters:
                parameter_values = self.df[parameter].tolist()[:i+1]
                min_log_close = min(parameter_values)
                max_log_close = max(parameter_values)
                normalized = [(p - min_log_close) / (max_log_close - min_log_close) for p in parameter_values]

                for s in normalized[-10:]:
                    state.append(s)

            state.append(1 if self.df['S_trend_d'].values[i] > 0 else 0)
            state.append(1 if self.last_bought_price > 0 else 0)  # add the info about holding position

            state = np.array(state, dtype=np.float32).flatten()  # 152 elements

            states.append(state)

        return np.array(states)   # states[0]  # np.array(states)  # states

    def step(self, action):
        # actions: 0 - hold, 1 - buy, 2 - sell

        reward = 0  # to motivate purchases
        profit = 0

        current_price = self.df['Close'].values[self.current_index]
        atr = self.df['ATR'].values[self.current_index]

        if self.number_of_shares_bought > 0:
            if self.df['Low'].values[self.current_index] < self.stop_loss:
                reward = -1

                # Sell by stop loss in 3% down
                self.last_bought_price = 0
                self.last_bought_position = 0
                self.number_of_shares_bought = 0

                self.sell_signals[self.current_index] = True

            elif self.df['High'].values[self.current_index] > self.take_profit:
                profit = (current_price - self.last_bought_price) / self.last_bought_price

                self.last_bought_price = 0
                self.last_bought_position = 0
                self.number_of_shares_bought = 0

                reward = 1

                self.sell_signals[self.current_index] = True

        if action == 1:
            if self.number_of_shares_bought == 0:
                number_of_shares = 100

                self.last_bought_price = current_price
                self.stop_loss = self.last_bought_price - 0.2 * atr
                self.take_profit = self.last_bought_price + atr
                self.last_bought_position = self.current_index
                self.number_of_shares_bought = number_of_shares
                self.total_deals += 1

                self.buy_signals[self.current_index] = True

                reward += 0.5

            reward += 0.01

        if action == 0:
            self.hold_signals[self.current_index] = True

            if self.number_of_shares_bought == 0:
                reward -= 0.3
            elif reward >= 0:
                reward += 1

        self.current_index += 1
        # (self.current_index > 100 and self.total_rewards < 3) or
        done = bool(self.current_index >= len(self.df['Close']) - PROFIT_PERIOD)
        self.done = done

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

        if self.mode != 'test':
            self.total_rewards += reward

        ACTIONS[action] += 1

        return self.get_state(), reward, done, {}

    def get_data(self, ticker):
        self.ticker = ticker

        self.df = get_data(ticker, period='day', days=700)

        self.prepare_data()
        # self.df.index = self.df.index.strftime('%m/%d/%Y %H:%M')

    def reset(self, cleanup_budget=True):

        print(self.total_rewards, self.current_index)

        if cleanup_budget:
            self.budget = INITIAL_DEPOSIT
        else:
            if self.last_bought_position > 0:
                self.budget += self.last_bought_price * self.number_of_shares_bought

        self.current_index = 60 + STATE_DATA  # we start from the Day #5 in the data set to have a history
        self.last_bought_position = 0
        self.last_bought_price = 0
        self.number_of_shares_bought = 0
        self.sum_profit = 0

        self.buy_signals = {i: False for i in range(len(self.df))}
        self.sell_signals = {i: False for i in range(len(self.df))}
        self.hold_signals = {i: False for i in range(len(self.df))}

        self.failures = 0
        self.total_rewards = 0
        self.done = False

        return self.get_state()

    def draw(self):
        df = self.df

        graph = make_subplots(rows=1, cols=1)
        graph.update_layout(title=self.ticker, xaxis_rangeslider_visible=False)

        graph.add_candlestick(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])

        graph.add_scatter(y=df['SMA50'], mode='lines', name='SMA50',
                          line={'color': 'magenta'})
        # graph.add_scatter(y=df['SMA200'], mode='lines', name='SMA200',
        #                   line={'color': 'blue', 'width': 3})

        buy_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['buy_signals'] == True]
        graph.add_scatter(x=buy_signals, y=df['Close'].values[buy_signals], name='Buy Signal', mode='markers',
                          marker_symbol='triangle-up', marker_color='#00FE35', marker_size=15, row=1, col=1)

        sell_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['sell_signals'] == True]
        graph.add_scatter(x=sell_signals, y=df['Close'].values[sell_signals], name='Sell Signal', mode='markers',
                          marker_symbol='triangle-down', marker_color='#D62728', marker_size=15, row=1, col=1)

        hold_signals = [i for i, (index, row) in enumerate(df.iterrows()) if row['hold_signals'] == True]
        graph.add_scatter(x=hold_signals, y=df['Close'].values[hold_signals], name='HOLD Signal', mode='markers',
                          marker_symbol='hash', marker_color='#555555', marker_size=15, row=1, col=1)

        graph.show()

    def render(self, mode='human', close=False):
        if self.total_deals > 0 and self.total_deals % 100 == 0:
            pass

        if self.total_rewards > 50:
            self.df['buy_signals'] = self.buy_signals.values()
            self.df['sell_signals'] = self.sell_signals.values()
            self.df['hold_signals'] = self.hold_signals.values()
            self.draw()

            exit()


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]

TICKERS.remove('GOOG')
TICKERS.remove('TSLA')

TICKERS = ['MSFT']
non_vectorized_env = ProfitEnv(TICKERS[:1].copy())  # :100
# wrapped_env = gym.wrappers.Monitor(non_vectorized_env, './results', force=True)
env = DummyVecEnv([lambda: non_vectorized_env])

state = env.reset()

# check_env(non_vectorized_env)

model = PPO('MlpPolicy', env, verbose=False, tensorboard_log='logs')  # learning_rate=0.0001
model.learn(total_timesteps=100000, progress_bar=True)


# non_vectorized_env = ProfitEnv(TICKERS[:1].copy(), mode='test')
# env = gym.wrappers.Monitor(non_vectorized_env, './results', force=True)
env.reset()
# env = DummyVecEnv([lambda: wrapped_env])

evaluate_policy(model, env, n_eval_episodes=1, render=True)
