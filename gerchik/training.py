import uuid
import os

from bot.utils import get_data
# from crypto_forex.utils import ALL_TICKERS

from tqdm import tqdm

import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio

from datetime import datetime
from datetime import timedelta

# TICKERS = sorted(ALL_TICKERS)[:1]
TICKERS = ['AMD']
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

    df['candle_size'] = df['High'] - df['Low']
    atr = df['candle_size'].mean()

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

    for daily_index, (date, selected_level) in markers:
        start = date.split(',')[0] + ', 10:00:00'
        start = datetime.strptime(start, "%Y-%m-%d, %H:%M:%S")
        end = date.split(',')[0] + ', 23:59:59'
        end = datetime.strptime(end, "%Y-%m-%d, %H:%M:%S")

        df_small_timeframe = get_data(ticker, period='minute', multiplier=5,
                                      start_date=start, end_date=end, save_data=False)

        # iterate over 5 minutes timeframe and find exact moment
        short_markers = []
        prev_date = None
        for i, (index, row) in enumerate(df_small_timeframe.iterrows()):
            if row['Low'] < selected_level < row['High']:
                if prev_date and i > 10:

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
        for small_timeframe_index, date_and_time in short_markers:
            case_id = str(uuid.uuid4())
            os.makedirs(f'training_data/{case_id}')

            end_moment = datetime.strptime(date_and_time, "%Y-%m-%d, %H:%M:%S")
            start_days = end_moment - timedelta(days=50)

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
            empty_rows = pd.DataFrame(index=range(10), columns=df_small_timeframe.columns)
            df_small_timeframe = pd.concat([df_small_timeframe, empty_rows])

            empty_rows = pd.DataFrame(index=range(2), columns=df.columns)
            df = pd.concat([df, empty_rows])

            graph = make_subplots(rows=1, cols=1, shared_xaxes=False,
                                  subplot_titles=['1 day timeframe'])
            graph.update_layout(title="", xaxis_rangeslider_visible=False,
                                xaxis=dict(showticklabels=False),
                                paper_bgcolor='white',
                                plot_bgcolor='white')

            graph.add_ohlc(x=df.index,
                           open=df['Open'],
                           high=df['High'],
                           low=df['Low'],
                           close=df['Close'],
                           decreasing={'line': {'color': 'black'}},
                           increasing={'line': {'color': 'black'}},
                           row=1, col=1, showlegend=False)

            graph.update_xaxes(showticklabels=False, row=1, col=1)
            graph.update_xaxes(rangeslider={'visible': False}, row=1, col=1)

            custom_ticks_daily = [round(level, 2) for level in levels_prices]
            custom_tick_text_daily = [str(value) for value in custom_ticks_daily]

            graph.update_layout(
                yaxis=dict(
                    tickvals=custom_ticks_daily,
                    ticktext=custom_tick_text_daily,
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                )
            )

            for t in RESULTS:
                for level_type in RESULTS[t]:
                    for level in RESULTS[t][level_type]:

                        graph.add_shape(type='line', x0=0, x1=len(df), y0=level, y1=level,
                                        line=dict(color='black', width=0.5),
                                        row=1, col=1)

            # Add bold line for selected level
            graph.add_shape(type='line', x0=0, x1=len(df),
                            y0=selected_level, y1=selected_level,
                            line=dict(color='black', width=3),
                            row=1, col=1)

            pio.write_image(graph, f'training_data/{case_id}/daily.png', height=1500, width=3000)

            # -------

            graph = make_subplots(rows=1, cols=1, shared_xaxes=False,
                                  subplot_titles=['5 minutes timeframe'])
            graph.update_layout(title="", xaxis_rangeslider_visible=False,
                                xaxis=dict(showticklabels=False),
                                paper_bgcolor='white',
                                plot_bgcolor='white')

            graph.add_ohlc(x=df_small_timeframe.index,
                           open=df_small_timeframe['Open'],
                           high=df_small_timeframe['High'],
                           low=df_small_timeframe['Low'],
                           close=df_small_timeframe['Close'],
                           decreasing={'line': {'color': 'black', 'width': 2}},
                           increasing={'line': {'color': 'black', 'width': 2}},
                           row=1, col=1, showlegend=False)

            graph.update_xaxes(showticklabels=False, row=1, col=1)
            graph.update_xaxes(rangeslider={'visible': False}, row=1, col=1)

            custom_ticks = [round(selected_level, 2),
                            df_small_timeframe['High'].max(),
                            df_small_timeframe['Low'].min(),
                            ]  # Add other default values as needed
            custom_tick_text = [str(value) for value in custom_ticks]
            graph.update_layout(
                yaxis=dict(
                    tickvals=custom_ticks,
                    ticktext=custom_tick_text,
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                )
            )

            graph.add_shape(type='line', x0=0, x1=len(df_small_timeframe),
                            y0=selected_level, y1=selected_level,
                            line=dict(color='black', width=3),
                            row=1, col=1)

            pio.write_image(graph, f'training_data/{case_id}/5_minutes.png', height=1500, width=3000)
