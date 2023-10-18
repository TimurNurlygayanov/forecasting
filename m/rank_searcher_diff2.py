
import numpy as np
import pandas as pd
import pandas_ta  # for TA magic

from tqdm import tqdm

from bot.utils import get_data
from bot.utils import get_tickers_polygon


RANK = {}
TICKERS = get_tickers_polygon(limit=5000)
SIGNALS = []
DATA_TO_CHECK = {}
CHECK_PERIOD = 40
MAX_STOP_LOSS = 0.8
RISK_REWARD_RATE = 3


def check_strategy_super_trend(df):
    intervals = []
    trends = []

    for i in [9, 21, 34, 50, 100, 150, 200]:
        df.ta.wma(length=i, append=True, col_names=(f'WMA{i}',))
        intervals.append(i)

    for i in [10, 34]:
        for j in [1.5, 2, 2.5, 3, 4]:
            df.ta.supertrend(append=True, length=i, multiplier=j,
                             col_names=(f'S_trend_{i}_{j}', f'S_trend_d_{i}_{j}', f'S_trend_l_{i}_{j}', f'S_trend_s_{i}_{j}',))
            trends.append(f'S_trend_d_{i}_{j}')

    df = df[200:].copy()
    last_index = df.shape[0] - 1

    if last_index < 100:
        return None

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    max_total_profit = [1000, 0, 0, 0, 0, 1000, "", 2, 0]

    for i1 in intervals:
        for t2 in trends:
            for stop_loss_factor in [1, 1.1, 1.5, 2]:
                ema1 = f'WMA{i1}'

                # Check if we got any signals for the last 3 days
                signal = False
                if df['Close'].values[last_index] > df[ema1].values[last_index]:
                    if df[t2].values[last_index - 1] < 0 < df[t2].values[last_index]:
                        signal = True

                        stop_loss_price_now = df['Close'].values[last_index] - stop_loss_factor * df['ATR'].values[
                            last_index]
                        take_profit_price_now = df['Close'].values[last_index] + 3 * stop_loss_factor * \
                                                df['ATR'].values[last_index]

                if not signal:
                    # no signal for now, no need to check the
                    # effectiveness of the strategy on history data
                    continue

                total_profit = 1000
                number_of_deals = 0
                good_deals = 0
                purchase_index = 0
                average_period = 0

                for i, (index, row) in enumerate(df.iterrows()):
                    stop_loss_percent = (row['Close'] - stop_loss_factor * row['ATR']) / row['Close']

                    if purchase_price == 0:
                        if row['Close'] > row[ema1] and df[t2].values[i-1] < 0 < row[t2]:
                            if stop_loss_percent > MAX_STOP_LOSS:
                                purchase_price = row['Close']
                                stop_loss_price = purchase_price - stop_loss_factor * row['ATR']
                                take_profit_price = purchase_price + 3 * stop_loss_factor * row['ATR']

                                purchase_index = i
                    else:
                        if row['Low'] < stop_loss_price:
                            total_profit = (stop_loss_price/purchase_price) * total_profit
                            purchase_price = 0
                            number_of_deals += 1

                            average_period += i - purchase_index

                        elif row['High'] > take_profit_price:
                            total_profit = (take_profit_price/purchase_price) * total_profit
                            purchase_price = 0
                            number_of_deals += 1
                            good_deals += 1

                            average_period += i - purchase_index

                average_period = int(average_period / number_of_deals) if number_of_deals > 0 else 1
                if max_total_profit[0] < total_profit and good_deals / number_of_deals > 0.45:
                    max_total_profit[0] = total_profit
                    max_total_profit[1] = ema1
                    max_total_profit[2] = t2
                    max_total_profit[3] = number_of_deals
                    max_total_profit[4] = good_deals
                    max_total_profit[5] = average_period
                    max_total_profit[6] = "Close higher than WMA and Super Trend started"
                    max_total_profit[7] = stop_loss_factor
                    max_total_profit[8] = (stop_loss_price_now, take_profit_price_now)

    if max_total_profit[0] > 1000:
        SIGNALS.append((ticker, max_total_profit))


def check_strategy_wma_crossover(df):
    intervals = []

    for i in range(3, 100, 2):
        df.ta.wma(length=i, append=True, col_names=(f'WMA{i}',))
        intervals.append(i)

    df = df[200:].copy()
    last_index = df.shape[0] - 1

    if last_index < 100:
        return None

    # if df['Close'].values[last_index] < df['EMA50_X'].values[last_index]:
    #     return None

    dfHA = df.ta.ha()
    dfHA.rename(columns={'HA_open': 'Open', 'HA_close': 'Close', 'HA_low': 'Low', 'HA_high': 'High'}, inplace=True)
    dfHA.ta.supertrend(append=True, length=34, multiplier=3.0,
                       col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    max_total_profit = [1000, 0, 0, 0, 0, 1000, "", 2, 0]

    for i1 in intervals:
        for i2 in intervals:
            if i1 >= i2:
                continue

            for stop_loss_factor in [1, 1.1, 1.5, 2]:
                wma1 = f'WMA{i1}'
                wma2 = f'WMA{i2}'

                if wma1 == wma2:
                    continue

                # Check if we got any signals for the last 3 days
                signal = False

                if df[wma1].values[last_index] > df[wma2].values[last_index]:
                    if df[wma1].values[last_index - 1] < df[wma2].values[last_index - 1]:
                        current_price = df['Close'].values[last_index]

                        # low = df['Low'].values[last_index]
                        # last_swing_low_now = min(df['Low'].values[last_index - 10:])
                        stop_loss_price_now = current_price - stop_loss_factor * df['ATR'].values[last_index]

                        if stop_loss_price_now / current_price > MAX_STOP_LOSS:
                            signal = True
                            take_profit_price_now = current_price + RISK_REWARD_RATE * abs(current_price - stop_loss_price_now)

                if not signal:
                    # no signal for now, no need to check the
                    # effectiveness of the strategy on history data
                    continue

                total_profit = 1000
                number_of_deals = 0
                good_deals = 0
                purchase_index = 0
                average_period = 0

                for i, (index, row) in enumerate(df.iterrows()):
                    if i < 10:
                        continue

                    if purchase_price == 0:
                        if row[wma1] > row[wma2] and df[wma1].values[i-1] < df[wma2].values[i-1]:

                            # get the lowest point from the past 20 candles:
                            # last_swing_low = min(df['Low'].values[i - 10:i])
                            stop_loss_price = row['Close'] - stop_loss_factor * row['ATR']

                            if stop_loss_price / row['Close'] > MAX_STOP_LOSS:
                                purchase_price = row['Close']

                                take_profit_price = purchase_price + RISK_REWARD_RATE * abs(
                                    purchase_price - stop_loss_price)

                                purchase_index = i
                    else:
                        if row['Low'] < stop_loss_price:
                            total_profit = (stop_loss_price/purchase_price) * total_profit
                            purchase_price = 0
                            number_of_deals += 1

                            average_period += i - purchase_index

                        elif row['High'] > take_profit_price:
                            total_profit = (take_profit_price/purchase_price) * total_profit
                            purchase_price = 0
                            number_of_deals += 1
                            good_deals += 1

                            average_period += i - purchase_index

                average_period = int(average_period / number_of_deals) if number_of_deals > 0 else 1
                if max_total_profit[0] < total_profit:
                    if good_deals / number_of_deals > 0.45:
                        if good_deals > 3 and average_period > 1:
                            max_total_profit[0] = total_profit
                            max_total_profit[1] = wma1
                            max_total_profit[2] = wma2
                            max_total_profit[3] = number_of_deals
                            max_total_profit[4] = good_deals
                            max_total_profit[5] = average_period
                            max_total_profit[6] = "WMA crossover"
                            max_total_profit[7] = stop_loss_factor
                            max_total_profit[8] = (stop_loss_price_now, take_profit_price_now)

    if max_total_profit[0] > 1000:
        SIGNALS.append((ticker, max_total_profit))


def check_strategy_macd(df):
    intervals = []

    for i in [9, 21, 34, 50, 100, 150, 200]:
        df.ta.ema(length=i, append=True, col_names=(f'EMA{i}',))
        intervals.append(i)

    df.ta.macd(append=True, col_names=('MACD', 'MACD_hist', 'MACD_signal'))

    df = df[200:].copy()
    last_index = df.shape[0] - 1

    if last_index < 100:
        return None

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    max_total_profit = [1000, 0, 0, 0, 0, 1000, "", 2, 0]

    for i1 in intervals:
        for stop_loss_factor in [0.1, 0.3, 0.5, 1, 1.1, 1.5, 2, 2.5, 3]:
            ema1 = f'EMA{i1}'

            signal = False
            if df['Close'].values[last_index] > df[ema1].values[last_index]:
                if df['MACD_hist'].values[last_index] > 0 > df['MACD_hist'].values[last_index - 1]:
                    if df['MACD'].values[last_index] < 0:
                        signal = True

                        stop_loss_price_now = df['Close'].values[last_index] - stop_loss_factor * df['ATR'].values[
                            last_index]
                        take_profit_price_now = df['Close'].values[last_index] + 3 * stop_loss_factor * \
                                                df['ATR'].values[last_index]


            if not signal:
                # no signal for now, no need to check the
                # effectiveness of the strategy on history data
                continue

            total_profit = 1000
            number_of_deals = 0
            good_deals = 0
            purchase_index = 0
            average_period = 0

            for i, (index, row) in enumerate(df.iterrows()):
                stop_loss_percent = (row['Close'] - stop_loss_factor * row['ATR']) / row['Close']

                if purchase_price == 0:
                    if row['Close'] > row[ema1]:
                        if row['MACD'] < 0:
                            if row['MACD_hist'] > 0 > df['MACD_hist'].values[i - 1]:
                                if stop_loss_percent > MAX_STOP_LOSS:
                                    purchase_price = row['Close']
                                    stop_loss_price = purchase_price - stop_loss_factor * row['ATR']
                                    take_profit_price = purchase_price + 3 * stop_loss_factor * row['ATR']

                                    purchase_index = i
                else:
                    if row['Low'] < stop_loss_price:
                        total_profit = (stop_loss_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1

                        average_period += i - purchase_index

                    elif row['High'] > take_profit_price:
                        total_profit = (take_profit_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1
                        good_deals += 1

                        average_period += i - purchase_index

            average_period = int(average_period / number_of_deals) if number_of_deals > 0 else 1
            if max_total_profit[0] < total_profit and good_deals / number_of_deals > 0.45:
                max_total_profit[0] = total_profit
                max_total_profit[1] = ema1
                max_total_profit[2] = 0
                max_total_profit[3] = number_of_deals
                max_total_profit[4] = good_deals
                max_total_profit[5] = average_period
                max_total_profit[6] = "Close higher than EMA and MACD crossover"
                max_total_profit[7] = stop_loss_factor
                max_total_profit[8] = (stop_loss_price_now, take_profit_price_now)

    if max_total_profit[0] > 1000:
        SIGNALS.append((ticker, max_total_profit))


def check_strategy_candles(df):

    df.ta.cdl_pattern(append=True, name=["morningstar", "hammer", "engulfing"])
    df.ta.ema(length=200, append=True, col_names=('EMA200',))

    df = df[200:].copy()
    last_index = df.shape[0] - 1

    if last_index < 100:
        return None

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    stop_loss_price_now = 0
    take_profit_price_now = 0
    max_total_profit = [1000, 0, 0, 0, 0, 1000, "", 2, 0]

    df['morningstar_and_hammer'] = df['CDL_MORNINGSTAR'] + df['CDL_HAMMER']
    df['morningstar_and_engulfing'] = df['CDL_MORNINGSTAR'] + df['CDL_ENGULFING']
    df['engulfing_and_hammer'] = df['CDL_ENGULFING'] + df['CDL_HAMMER']
    df['all_candle_patterns'] = df['CDL_MORNINGSTAR'] + df['CDL_HAMMER'] + df['CDL_ENGULFING']
    patterns = ['CDL_MORNINGSTAR', 'CDL_HAMMER', 'CDL_ENGULFING', 'morningstar_and_hammer',
                'morningstar_and_engulfing', 'engulfing_and_hammer', 'all_candle_patterns']

    for p in patterns:
        for stop_loss_factor in [0.1, 0.3, 0.5, 1, 1.1, 1.5, 2, 2.5, 3]:
            signal = False
            if df['Close'].values[last_index] > df['EMA200'].values[last_index]:
                if df[p].values[last_index] > 0:
                    signal = True
                    stop_loss_price_now = df['Close'].values[last_index] - stop_loss_factor * df['ATR'].values[last_index]
                    take_profit_price_now = df['Close'].values[last_index] + 3 * stop_loss_factor * df['ATR'].values[last_index]

            if not signal:
                # no signal for now, no need to check the
                # effectiveness of the strategy on history data
                continue

            total_profit = 1000
            number_of_deals = 0
            good_deals = 0
            purchase_index = 0
            average_period = 0

            for i, (index, row) in enumerate(df.iterrows()):
                stop_loss_percent = (row['Close'] - stop_loss_factor * row['ATR']) / row['Close']

                if purchase_price == 0:
                    if row['Close'] > row['EMA200']:
                        if row[p] > 0:
                            if stop_loss_percent > MAX_STOP_LOSS:
                                purchase_price = row['Close']
                                stop_loss_price = purchase_price - stop_loss_factor * row['ATR']
                                take_profit_price = purchase_price + 3 * stop_loss_factor * row['ATR']

                                purchase_index = i
                else:
                    if row['Low'] < stop_loss_price:
                        total_profit = (stop_loss_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1

                        average_period += i - purchase_index

                    elif row['High'] > take_profit_price:
                        total_profit = (take_profit_price/purchase_price) * total_profit
                        purchase_price = 0
                        number_of_deals += 1
                        good_deals += 1

                        average_period += i - purchase_index

            average_period = average_period / number_of_deals if number_of_deals > 0 else 1
            if max_total_profit[0] < total_profit:
                if good_deals / number_of_deals > 0.45:
                    if good_deals > 3:
                        max_total_profit[0] = total_profit
                        max_total_profit[1] = p
                        max_total_profit[2] = 0
                        max_total_profit[3] = number_of_deals
                        max_total_profit[4] = good_deals
                        max_total_profit[5] = average_period
                        max_total_profit[6] = p
                        max_total_profit[7] = stop_loss_factor
                        max_total_profit[8] = (stop_loss_price_now, take_profit_price_now, purchase_price)


    if max_total_profit[0] > 1000:
        SIGNALS.append((ticker, max_total_profit))


def check_strategy_ha(df):
    last_index = df.shape[0] - 1

    if last_index < 100:
        return None

    dfHA = df.ta.ha()
    dfHA.rename(columns={'HA_open': 'Open', 'HA_close': 'Close', 'HA_low': 'Low', 'HA_high': 'High'}, inplace=True)
    dfHA.ta.supertrend(append=True, length=34, multiplier=3.0,
                       col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
    df.ta.supertrend(append=True, length=10, multiplier=2,
                     col_names=('S_trend', 'S_trend_d', 'S_trend_l', 'S_trend_s',))
    dfHA.ta.ema(length=21, append=True, col_names=('WMA21',))

    purchase_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    max_total_profit = [1000, 0, 0, 0, 0, 1000, "", 2, 0]

    for stop_loss_factor in [1, 1.1, 1.5, 2]:
        # Check if we got any signals for the last 3 days
        signal = False

        if dfHA['Open'].values[last_index] == dfHA['Low'].values[last_index]:
            if dfHA['Close'].values[last_index] > dfHA['WMA21'].values[last_index] > dfHA['Open'].values[last_index]:
                if dfHA['S_trend_d'].values[last_index] > 0:
                    # if df['S_trend_d'].values[last_index] > 0 > df['S_trend_d'].values[last_index - 1]:

                    current_price = df['Close'].values[last_index]

                    # low = df['Low'].values[last_index]
                    # last_swing_low_now = min(df['Low'].values[last_index - 10:])
                    stop_loss_price_now = current_price - stop_loss_factor * df['ATR'].values[last_index]

                    if stop_loss_price_now / current_price > MAX_STOP_LOSS:
                        signal = True
                        take_profit_price_now = current_price + RISK_REWARD_RATE * abs(current_price - stop_loss_price_now)

        if not signal:
            # no signal for now, no need to check the
            # effectiveness of the strategy on history data
            continue

        total_profit = 1000
        number_of_deals = 0
        good_deals = 0
        purchase_index = 0
        average_period = 0

        for i, (index, row) in enumerate(df.iterrows()):
            if i < 10:
                continue

            if purchase_price == 0:
                if dfHA['Open'].values[i] == dfHA['Low'].values[i]:
                    if dfHA['Close'].values[i] > dfHA['WMA21'].values[i] > dfHA['Open'].values[i]:
                        if dfHA['S_trend_d'].values[i] > 0:
                            # get the lowest point from the past 20 candles:
                            # last_swing_low = min(df['Low'].values[i - 10:i])
                            stop_loss_price = row['Close'] - stop_loss_factor * row['ATR']

                            if stop_loss_price / row['Close'] > MAX_STOP_LOSS:
                                purchase_price = row['Close']

                                take_profit_price = purchase_price + RISK_REWARD_RATE * abs(
                                    purchase_price - stop_loss_price)

                                purchase_index = i
            else:
                if row['Low'] < stop_loss_price:
                    total_profit = (stop_loss_price/purchase_price) * total_profit
                    purchase_price = 0
                    number_of_deals += 1

                    average_period += i - purchase_index

                elif row['High'] > take_profit_price:
                    total_profit = (take_profit_price/purchase_price) * total_profit
                    purchase_price = 0
                    number_of_deals += 1
                    good_deals += 1

                    average_period += i - purchase_index

        average_period = int(average_period / number_of_deals) if number_of_deals > 0 else 1
        if max_total_profit[0] < total_profit:
            # print(ticker, total_profit, number_of_deals, good_deals / number_of_deals)
            if good_deals / number_of_deals > 0.45:
                if good_deals > 3:
                    max_total_profit[0] = total_profit
                    max_total_profit[1] = 0
                    max_total_profit[2] = 0
                    max_total_profit[3] = number_of_deals
                    max_total_profit[4] = good_deals
                    max_total_profit[5] = average_period
                    max_total_profit[6] = "Heikin Ashi"
                    max_total_profit[7] = stop_loss_factor
                    max_total_profit[8] = (stop_loss_price_now, take_profit_price_now)

    if max_total_profit[0] > 1000:
        SIGNALS.append((ticker, max_total_profit))


if __name__ == '__main__':

    date_printed = False

    for ticker in tqdm(TICKERS[:1000]):
        df = get_data(ticker)

        if CHECK_PERIOD > 0:
            DATA_TO_CHECK[ticker] = df.tail(CHECK_PERIOD)
            df = df.iloc[:-CHECK_PERIOD]

        if not date_printed:
            print(f'Date of trade: {df.iloc[-1].name}')
            date_printed = True

        df.ta.atr(append=True, col_names=('ATR',))
        df.ta.ema(length=50, append=True, col_names=('EMA50_X',))

        # check_strategy_candles(df.copy())
        # check_strategy_super_trend(df.copy())
        # check_strategy_wma_crossover(df.copy())
        # check_strategy_macd(df.copy())
        check_strategy_ha(df.copy())

    if len(SIGNALS) == 0:
        print('No signals detected!')
        exit(1)

    # Remove all signals that lead to very long waiting
    expected_time_medium = 1.1 * sum([s[1][5] for s in SIGNALS]) / len(SIGNALS)
    SIGNALS = [s for s in SIGNALS if s[1][5] < expected_time_medium]

    # Remove signals that give less than 45% of success rate
    SIGNALS = [s for s in SIGNALS if s[1][4] / s[1][3] > 0.45]

    total_good_deals = 0
    deals_to_check = min(20, len(SIGNALS))

    for s in sorted(SIGNALS, key=lambda x: x[1][0], reverse=True)[:deals_to_check]:
        print(f"{s[0]}: win rate {100*s[1][4] / s[1][3]:.1f}, period {s[1][5]}"
              f" stop loss & take profit: {s[1][8][0]:.2f} {s[1][8][1]:.2f}")

        if CHECK_PERIOD > 0:
            status = 0
            for i, (index, row) in enumerate(DATA_TO_CHECK[s[0]].iterrows()):
                if status == 0:
                    if row['Low'] < s[1][8][0]:
                        print('Failed!', s)
                        status = -1
                    elif row['High'] > s[1][8][1]:
                        print('Profit!', s)
                        status = 1

            if status > 0:
                total_good_deals += 1
            elif status == 0:
                print('Not ready yet', s)

        print('- ' * 20)
        print()

    print(f'WIN RATE TOTAL: {100 * total_good_deals / deals_to_check:.1f} %')
