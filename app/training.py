
import os

from bot.utils import get_data
from crypto_forex.utils import ALL_TICKERS

from tqdm import tqdm
import hashlib

import pandas as pd

from datetime import datetime
from datetime import timedelta

from utils import draw


TICKERS = sorted(ALL_TICKERS)[:1]
# TICKERS = ['AMD']
RESULTS = {}


for ticker in tqdm(TICKERS):
    df = get_data(ticker, period='day', multiplier=1, days=100, save_data=False)
    df.sort_index(inplace=True)

    if df is None or df.empty or len(df) < 20:
        continue

    price_diff = 0.1

    RESULTS[ticker] = {'limit': set(), 'mirror': set(), 'trend_reversal': set()}
    levels_found = 0
    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    # find local mins and maxs
    df['minima'] = df['Low'] == df['Low'].rolling(window=20, center=True).min()
    df['maxima'] = df['High'] == df['High'].rolling(window=20, center=True).max()
    mins = df[df['minima']]['Low'].tolist()
    maxs = df[df['maxima']]['High'].tolist()

    df_short = df.iloc[-30:].copy()
    df_short['candle_size'] = df_short['High'] - df_short['Low']
    candle_sizes = sorted(df_short['candle_size'].tolist())[10:-10]  # get rid of large and small candles
    atr = sum(candle_sizes) / len(candle_sizes)

    for price_level in mins + maxs:
        RESULTS[ticker]['trend_reversal'].add(price_level)
        levels_found += 1

    for level_type in ['limit', 'mirror']:
        prices = highs + lows
        bars_required = 2

        if level_type == 'mirror':
            prices = sorted(prices)
            bars_required = 3

        group = []
        limit_levels = set()
        previous_price = prices[0]

        for p in prices:
            if 100 * abs(previous_price - p) / p < price_diff:
                group.append(p)
            else:
                if len(group) >= bars_required:
                    level = sum(group) / len(group)

                    limit_levels.add(level)
                    levels_found += 1

                group = []

            previous_price = p

        if limit_levels:
            RESULTS[ticker][level_type] = limit_levels

    if not levels_found:
        del RESULTS[ticker]

    # find all levels:
    levels_prices = []
    for t in RESULTS:
        for level_type in RESULTS[t]:
            for level in RESULTS[t][level_type]:
                levels_prices.append(level)

    # iterate over data and find dates when the price was crossing the level:
    markers = []
    for i, (index, row) in enumerate(df.iterrows()):
        for level in levels_prices:
            if row['Low'] < level < row['High']:
                markers.append((i, (index, level)))

    for daily_index, (date, selected_level) in set(markers):
        date_str = date.strftime("%Y-%m-%d, %H:%M:%S")
        start = date_str.split(',')[0] + ', 10:00:00'
        start = datetime.strptime(start, "%Y-%m-%d, %H:%M:%S")
        end = date_str.split(',')[0] + ', 21:00:00'
        end = datetime.strptime(end, "%Y-%m-%d, %H:%M:%S")

        df_small_timeframe = get_data(ticker, period='minute', multiplier=5,
                                      start_date=start, end_date=end, save_data=False)

        # iterate over 5 minutes timeframe and find exact moment
        short_markers = []
        prev_date = None
        for i, (index, row) in enumerate(df_small_timeframe.iterrows()):
            if row['Low'] < selected_level < row['High']:
                if prev_date and i > 20:

                    # do not include level if we crossed ir withing last 10 small candles
                    skip_level = False
                    for k in range(i-10, i):
                        candle_open = df_small_timeframe.iloc[k]['Open']
                        candle_close = df_small_timeframe.iloc[k]['Close']

                        if candle_open > selected_level > candle_close or \
                                candle_open < selected_level < candle_close:
                            skip_level = True

                    distance1 = abs(df_small_timeframe.iloc[i-1]['Low'] - selected_level)
                    distance2 = abs(df_small_timeframe.iloc[i-1]['High'] - selected_level)
                    distance = min(distance1, distance2)

                    # do not include level is we closed far away from it
                    if distance < 0.1 * atr and not skip_level:
                        short_markers.append((i, prev_date))

            prev_date = index

        # TODO: make short_markers shorter, like only 3 points (start, end, middle)
        for small_timeframe_index, date_and_time in set(short_markers):
            end_moment = date_and_time
            start_days = end_moment - timedelta(days=100)

            case_id = hashlib.md5((ticker + str(selected_level) + str(end_moment)).encode()).hexdigest()
            os.makedirs(f'training_data/{case_id}')

            # calculate level for stop loss - TVH
            profit_factor = 3
            tvh1 = selected_level + (0.2 * atr) * 0.1
            level_stop = selected_level - (0.2 * atr) * 0.9
            take_profit1 = tvh1 + profit_factor * 0.2 * atr

            tvh2 = selected_level - (0.2 * atr) * 0.1
            level_stop2 = selected_level + (0.2 * atr) * 0.9
            take_profit2 = tvh2 - profit_factor * 0.2 * atr

            df_small_timeframe = get_data(ticker, period='minute', multiplier=5,
                                          start_date=start,
                                          end_date=end_moment,
                                          save_data=False)

            df = get_data(ticker, period='day', multiplier=1,
                          start_date=start_days,
                          end_date=end_moment,
                          save_data=False)

            # Cut last daily to hide spoilers
            df.iloc[-1]['High'] = df_small_timeframe['High'].max()
            df.iloc[-1]['Low'] = df_small_timeframe['Low'].min()
            df.iloc[-1]['Close'] = df_small_timeframe.iloc[-1]['Close']
            df.iloc[-1]['Open'] = df_small_timeframe.iloc[0]['Open']

            # Add empty data to show it later:
            date_range = pd.date_range(
                end_moment + timedelta(minutes=5),
                periods=30, freq='5T'
            )
            # date_strings = date_range.strftime('%Y-%m-%d, %H:%M:%S')

            empty_rows = pd.DataFrame(index=date_range,
                                      columns=df_small_timeframe.columns)
            empty_rows['Open'] = selected_level
            empty_rows['Close'] = selected_level
            empty_rows['Low'] = selected_level
            empty_rows['High'] = selected_level

            df_small_timeframe = pd.concat([df_small_timeframe, empty_rows])

            date_range = pd.date_range(
                end_moment + timedelta(days=2),
                periods=2, freq='1D'
            )
            empty_rows = pd.DataFrame(index=date_range, columns=df.columns)
            empty_rows['Open'] = selected_level
            empty_rows['Close'] = selected_level
            empty_rows['Low'] = selected_level
            empty_rows['High'] = selected_level
            df = pd.concat([df, empty_rows])

            draw(df, case_id=case_id, custom_ticks=levels_prices, file_name='daily', selected_level=selected_level)

            # =---- get hourly data too

            df_hourly = get_data(ticker, period='hour', multiplier=1,
                                 start_date=end_moment - timedelta(days=3),
                                 end_date=end_moment,
                                 save_data=False)

            latest_date = df_hourly.index.max().strftime("%Y-%m-%d, %H:%M:%S")
            start_latest_date = ':'.join(latest_date.split(':')[:-2]) + ':00:00'
            start_latest_date = datetime.strptime(start_latest_date, "%Y-%m-%d, %H:%M:%S")

            df_last_hour = get_data(ticker, period='minute', multiplier=5,
                                    start_date=start_latest_date,
                                    end_date=end_moment,
                                    save_data=False)

            # Cut last hourly to hide spoilers
            df_hourly.iloc[-1]['High'] = df_last_hour['High'].max()
            df_hourly.iloc[-1]['Low'] = df_last_hour['Low'].min()
            df_hourly.iloc[-1]['Close'] = df_last_hour.iloc[-1]['Close']
            df_hourly.iloc[-1]['Open'] = df_last_hour.iloc[0]['Open']

            custom_ticks = [
                round(selected_level, 2),
                df_hourly['High'].max(),
                df_hourly['Low'].min(),
            ]
            draw(df_hourly, case_id=case_id, custom_ticks=custom_ticks, file_name='hourly', selected_level=selected_level)

            # -------

            custom_ticks = [round(selected_level, 2),
                            round(level_stop, 2),
                            round(level_stop2, 2),
                            round(take_profit1, 2),
                            round(take_profit2, 2),
                            df_small_timeframe['High'].max(),
                            df_small_timeframe['Low'].min(),
                            ]  # Add other default values as needed

            draw(df_small_timeframe, case_id=case_id, custom_ticks=custom_ticks, file_name='5_minutes', selected_level=selected_level)

            # Save data for the future use
            df.to_csv(f'training_data/{case_id}/daily_before.csv', index=True)
            df_small_timeframe.to_csv(f'training_data/{case_id}/5_minutes_before.csv', index=True)

            df_small_timeframe_after = get_data(
                ticker,
                period='minute', multiplier=5,
                start_date=start,
                end_date=end_moment + timedelta(minutes=5*30),
                save_data=False
            )

            df_small_timeframe_after.to_csv(
                f'training_data/{case_id}/5_minutes_after.csv', index=True
            )

            draw(df_small_timeframe_after, case_id=case_id, custom_ticks=custom_ticks, file_name='5_minutes_after', selected_level=selected_level)

            # =======

            df_short = df_small_timeframe.copy()
            df_short['candle_size'] = df_short['High'] - df_short['Low']

            medium_candle_size = 0
            if len(df_short) > 30:
                # get rid of large and small candles
                candle_sizes = sorted([s for s in df_short['candle_size'].tolist() if s > 0])[10:-10]

                if candle_sizes:
                    medium_candle_size = 1.5 * sum(candle_sizes) / len(candle_sizes)

            if medium_candle_size == 0:
                medium_candle_size = 1.5 * df_short['candle_size'].mean()

            deal = f"""[GLOBAL]
            atr={round(atr, 2)}
            level={round(selected_level, 2)}
            [DEAL1]
            tvh={round(tvh1, 2)}
            stop_loss={round(level_stop, 2)}
            take_profit={round(take_profit1, 2)}
            [DEAL2]
            tvh={round(tvh2, 2)}
            stop_loss={round(level_stop2, 2)}
            take_profit={round(take_profit2, 2)}
            [DEAL3]
            tvh={round(tvh1, 2)}
            stop_loss={round(selected_level - atr * 0.02, 2)}
            take_profit={round(tvh1 + 5 * atr * 0.02, 2)}
            [DEAL4]
            tvh={round(tvh2, 2)}
            stop_loss={round(selected_level + atr * 0.02, 2)}
            take_profit={round(tvh2 - 5 * atr * 0.02, 2)}
            [DEAL5]
            tvh={round(tvh2, 2)}
            stop_loss={round(tvh2 + medium_candle_size, 2)}
            take_profit={round(tvh2 - 3 * medium_candle_size, 2)}
            [DEAL6]
            tvh={round(tvh1, 2)}
            stop_loss={round(tvh1 - medium_candle_size, 2)}
            take_profit={round(tvh1 + 3 * medium_candle_size, 2)}
            """
            with open(f'training_data/{case_id}/deal.ini',
                      encoding='utf8', mode='w+') as f:
                f.writelines('\n'.join([line.strip() for line in deal.split()]))
