import numpy as np
import pandas as pd
import pandas_ta  # for TA magic
from os import path

from catboost import Pool
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


with open('smp500.txt', 'r') as f:
    TICKERS = f.readlines()

TICKERS = [t.replace('\n', '') for t in TICKERS if '^' not in t and '/' not in t and '.' not in t]


class DataCollector:

    data = {}
    periods = None
    labels = None

    learn_period = 5
    profit_period = 5
    take_profit_threshold = 1.10

    def __init__(self, tickers: list, period: int):

        for i, ticker in enumerate(tickers):
            df = pd.DataFrame()
            file_name = f'smart_predictor/data/{ticker}.xlsx'

            print(f'Reading {ticker}...', end='\r')

            if path.exists(file_name):
                df = pd.read_excel(file_name, index_col=0)  # , header=[0, 1]
                print(df.head())
            else:
                df = df.ta.ticker(ticker, period=f'{period+400}d')  # add extra 200 days here to calculate 200 SMA

                if df is None or len(df) < period+400:  # remove all tickers that don't have required data
                    continue

                df.ta.supertrend(length=10, multiplier=4.0, append=True,
                                 col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

                df.ta.sma(length=200, append=True, col_names=('SMA200',))
                df.ta.sma(length=50, append=True, col_names=('SMA50',))
                df.ta.sma(length=7, append=True, col_names=('SMA7',))

                df.ta.rsi(append=True, col_names=('RSI',))
                df.ta.bbands(col_names=('L', 'M', 'U', 'B', 'P'), append=True)

                df = df[200:-200].copy()  # cut last 200 days because we will test the model on the latest 200 days

                df.index = df.index.strftime('%m/%d/%Y')
                df.to_excel(f'smart_predictor/data/{ticker}.xlsx', index=True, header=True)

            self.data[ticker] = df

    def learn(self):
        self.periods = []
        self.labels = []

        # Date shouldn't be a feature, we exclude 0
        cat_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        # Specify the training parameters:
        model = CatBoostClassifier(iterations=1000, thread_count=7, random_seed=42,
                                   cat_features=cat_features,
                                   loss_function='Logloss', custom_loss=['AUC', 'Accuracy'])

        for ticker in self.data:
            result = {}

            for i, row in enumerate(self.data[ticker].iterrows()):
                result[i] = False

                max_profit_price = max(self.data[ticker]['Close'].values[i:i+self.profit_period])

                if row[1]['Close'] * self.take_profit_threshold < max_profit_price:
                    result[i] = True

            self.data[ticker] = self.data[ticker].applymap(str)
            self.data[ticker]['ACTION'] = result.values()

            X = self.data[ticker].drop("ACTION", axis=1)
            y = self.data[ticker]["ACTION"]

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

            model.fit(X_train, y_train, cat_features=cat_features, plot=True, eval_set=(X_val, y_val))


collector = DataCollector(TICKERS[:20], period=250*3)
collector.learn()
