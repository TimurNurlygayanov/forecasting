import asyncio
from telegram import Bot

import pandas as pd
import numpy as np
from typing import Tuple

from joblib import Parallel, delayed

from crypto_forex.utils import ALL_TICKERS
from crypto_forex.utils import get_data


with open('/Users/timur.nurlygaianov/telegram_token') as f:
    bot_token = ''.join(f.readlines()).strip()

chat_id = "335442091"


def send_message(message_text):
    bot = Bot(token=bot_token)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.send_message(chat_id=chat_id, text=message_text))


def will_frac(df: pd.DataFrame, period: int = 2) -> Tuple[pd.Series, pd.Series]:

    periods = [p for p in range(-period, period + 1) if p != 0]  # default [-2, -1, 1, 2]

    highs = [df['High'] > df['High'].shift(p) for p in periods]
    bears = pd.Series(np.logical_and.reduce(highs), index=df.index)

    lows = [df['Low'] < df['Low'].shift(p) for p in periods]
    bulls = pd.Series(np.logical_and.reduce(lows), index=df.index)

    return bears, bulls


def run_bot():
    send_message('= ' * 30)

    def run_me(ticker):
        print(ticker)

        for mult in [3, 5]:
            df = get_data(ticker, save_data=False, period='minute', multiplier=5, days=1)

            """
            df.ta.supertrend(append=True, length=10, multiplier=1,
                             col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
            df.ta.supertrend(append=True, length=11, multiplier=1.5,
                             col_names=('S_trend2', 'S_trend_d2', 'S_trend_l2', 'S_trend_s2',))
            df.ta.supertrend(append=True, length=34, multiplier=2,
                             col_names=('S_trend3', 'S_trend_d3', 'S_trend_l3', 'S_trend_s3',))
    
            df['signal'] = (df['S_trend_d'] == df['S_trend_d2']) & (df['S_trend_d2'] == df['S_trend_d3'])
            if df['signal'].values[-1]:
                if df['S_trend_d'].values[-1] > 0:
                    if df['S_trend'].values[-1] > df['S_trend'].values[-2]:
                        send_message(f"{ticker} buy by 3 super trends!")
            """

            dfHA = df.ta.ha()
            dfHA.rename(columns={'HA_open': 'Open', 'HA_close': 'Close', 'HA_low': 'Low', 'HA_high': 'High'}, inplace=True)
            df.ta.ema(length=21, append=True, col_names=('EMA21',))

            """
            if df['Close'].values[-1] > df['EMA21'].values[-1] > df['Open'].values[-1]:
                if df['EMA21'].values[-1] > df['EMA21'].values[-2]:
                    send_message(f"{ticker} closed on top of EMA 21!")
            if df['Close'].values[-1] < df['EMA21'].values[-1] < df['Open'].values[-1]:
                if df['EMA21'].values[-1] < df['EMA21'].values[-2]:
                    send_message(f"{ticker} closed lower than EMA 21!")
            """

            bears, bulls = will_frac(df, period=2)
            if bears[-2] or bears[-3]:
                send_message(f"{ticker} Williams signal - {mult} m!")
            if bulls[-2] or bulls[-3]:
                send_message(f"{ticker} Williams signal - {mult} m!")

            # print(bulls)

            """
            if dfHA['Close'].values[-1] > df['EMA21'].values[-1] > dfHA['Open'].values[-1]:
                if df['EMA21'].values[-1] > df['EMA21'].values[-2]:
                    send_message(f"{ticker} HA closed on top of EMA 21!")
            if dfHA['Close'].values[-1] < df['EMA21'].values[-1] < dfHA['Open'].values[-1]:
                if df['EMA21'].values[-1] < df['EMA21'].values[-2]:
                    send_message(f"{ticker} HA closed lower than EMA 21!")
            """

    Parallel(n_jobs=-1)(delayed(run_me)(ticker) for ticker in ALL_TICKERS)


if __name__ == "__main__":
    run_bot()
