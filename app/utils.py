import json
import requests
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder
import asyncio
from datetime import datetime
from datetime import timedelta
import pandas as pd

from forex_python.converter import CurrencyRates

from metaapi_cloud_sdk import MetaApi


async def initialize_connection():
    with open('/Users/timur.nurlygaianov/fx_api_token.txt', mode='r', encoding='utf8') as f:
        api_token = f.read().strip()

    api = MetaApi(token=api_token)
    account = await api.metatrader_account_api.get_account(account_id='46c858ce-aeed-47fd-a9f1-d4f8c5576ac8')
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    return account, connection


def get_data_alpha(ticker, interval='15min', limit=100):  # Daily
    with open('/Users/timur.nurlygaianov/alpha_key.txt', mode='r', encoding='utf8') as f:
        api_token = f.read().strip()

    ticker1 = ticker[:3]
    ticker2 = ticker[3:]

    data_type = 'FX_INTRADAY'
    if interval == 'Daily':
        data_type = 'FX_DAILY'

    outputsize = 'full'
    if limit <= 100:
        outputsize = 'compact'

    url = (f'https://www.alphavantage.co/query?function={data_type}&'
           f'from_symbol={ticker1}&to_symbol={ticker2}&interval={interval}&apikey={api_token}&outputsize={outputsize}')
    r = requests.get(url)
    data = r.json()

    indexes = []
    pd_data = {'Close': [], 'Open': [], 'Low': [], 'High': []}

    for date, values in list(reversed(data[f'Time Series FX ({interval})'].items()))[-limit:]:
        if len(indexes) < limit:
            indexes.append(date)

            pd_data['Open'].append(float(values['1. open']))
            pd_data['Close'].append(float(values['4. close']))
            pd_data['High'].append(float(values['2. high']))
            pd_data['Low'].append(float(values['3. low']))

    df = pd.DataFrame(pd_data, index=indexes)
    df.sort_index()

    return df


def get_atr(df):
    df_short = df.iloc[-50:].copy()
    df_short['candle_size'] = df_short['High'] - df_short['Low']
    candle_sizes = sorted(df_short['candle_size'].tolist())[10:-10]  # get rid of large and small candles
    atr = sum(candle_sizes) / len(candle_sizes)
    return atr


def draw(df, case_id, custom_ticks, file_name='', selected_level=0, price_diff=0.0001,
         horizontal_volumes=None):
    start_moment = df.index.min()
    end_moment = df.index.max()

    graph = make_subplots(rows=1, cols=1, shared_xaxes=False,
                          subplot_titles=[''])
    graph.update_layout(title="", xaxis_rangeslider_visible=False,
                        xaxis=dict(showticklabels=False),
                        paper_bgcolor='white',
                        plot_bgcolor='white')

    graph.add_ohlc(x=df.index,
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'],
                   decreasing={'line': {'color': 'black', 'width': 2}},
                   increasing={'line': {'color': 'black', 'width': 2}},
                   row=1, col=1, showlegend=False)

    graph.update_xaxes(showticklabels=False, row=1, col=1)
    graph.update_xaxes(rangeslider={'visible': False}, row=1, col=1)
    graph.update_xaxes(type='category', categoryorder='trace')  # to ignore missed dates

    # Filter daily levels to get rid of duplicates
    prev_level = 0
    levels_prices_filtered = []

    for level in sorted(custom_ticks):
        # if level == selected_level or abs(prev_level - level) > price_diff:
        levels_prices_filtered.append(level)

        prev_level = level

    custom_ticks_clear = [round(level, 6) for level in levels_prices_filtered]

    price_movement = (df['High'].max() - df['Low'].min()) / 2
    for level in levels_prices_filtered:
        if df['Low'].min() - price_movement < level < df['High'].max() + price_movement:
            graph.add_shape(type='line', x0=start_moment, x1=end_moment, y0=level, y1=level,
                            line=dict(color='black', width=1),
                            row=1, col=1)

    custom_ticks_text = [str(value) for value in custom_ticks_clear]

    graph.update_layout(
        yaxis=dict(
            tickvals=custom_ticks_clear,
            ticktext=custom_ticks_text,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickfont=dict(size=18)
        ),
        margin=dict(l=1, r=1, t=1, b=1),
        autosize=True
    )

    # Add bold line for selected level
    if selected_level:
        graph.add_shape(type='line', x0=start_moment, x1=end_moment,
                        y0=round(selected_level, 6), y1=round(selected_level, 6),
                        line=dict(color='black', width=4),
                        row=1, col=1)

    # Add horizontal volume
    if horizontal_volumes:
        delta = (df['High'].max() - df['Low'].min()) / 100  # split all price track to 100 small levels

        volume_delta = max(horizontal_volumes.values()) / len(df)
        dates = df.index.tolist()

        graph.add_bar(
            y=[df['Low'].min() + k * delta for k in range(0, 101)],
            x=[dates[int(v / volume_delta)-1] for v in list(horizontal_volumes.values())],
            orientation='h',
            marker=dict(color='yellow', opacity=0.2),
            row=1, col=1
        )

    pio.write_image(
        graph, f'training_data/{case_id}/{file_name}.png',
        height=1000, width=2000
    )

    with open(f'training_data/{case_id}/{file_name}.json', 'w+') as f:
        json.dump(graph, cls=PlotlyJSONEncoder, fp=f)


def get_levels(df, price_diff=0.001):
    """ This function automatically detects levels on daily timeframe. """

    if df is None or df.empty or len(df) < 20:
        return [], 0

    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    current_price = df['Close'].iloc[-1]
    last_low = df['Low'].iloc[-1]
    last_high = df['High'].iloc[-1]

    # find local mins and maxs
    k_min = df['Low'] == df['Low'].rolling(window=20, center=True).min()
    k_max = df['High'] == df['High'].rolling(window=20, center=True).max()
    mins = df[k_min]['Low'].tolist()
    maxs = df[k_max]['High'].tolist()

    levels_prices = [df['High'].max(), df['Low'].min()]
    for price_level in mins + maxs:
        levels_prices.append(price_level)

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
                    level = group[0]

                    limit_levels.add(level)

                group = []

            previous_price = p

        levels_prices += [level for level in limit_levels]

    selected_level = 0
    nearest_level_distance = current_price
    for level in set(levels_prices):
        if abs(current_price - level) < nearest_level_distance:
            selected_level = level
            nearest_level_distance = abs(current_price - level)

        if abs(last_low - level) < nearest_level_distance:
            selected_level = level
            nearest_level_distance = abs(current_price - level)

        if abs(last_high - level) < nearest_level_distance:
            selected_level = level
            nearest_level_distance = abs(current_price - level)

    return set(levels_prices), round(selected_level, 6)


def get_data_meta(ticker='EURUSD', timeframe='1d', limit=100):
    # ONLY FOR FOREX
    now = datetime.now()

    async def get_data_x():
        account, connection = await initialize_connection()

        candles = await account.get_historical_candles(symbol=ticker, timeframe=timeframe,
                                                       start_time=now, limit=limit)

        data = []
        for k in candles:
            date = k['brokerTime'].split('.')[0]
            date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

            data.append({
                'date': date,
                'Close': k['close'],
                'Open': k['open'],
                'High': k['high'],
                'Low': k['low']
            })

        df = pd.DataFrame(data)
        df = df.set_index('date')

        return df

    return asyncio.run(get_data_x())


def add_empty_rows(df, delta_minutes=5, length=30, selected_level=0):
    end_moment = df.index.max()

    # Add empty data to show it later:
    date_range = pd.date_range(
        end_moment + timedelta(minutes=delta_minutes),
        periods=length, freq='5T'
    )

    empty_rows = pd.DataFrame(index=date_range,
                              columns=df.columns)
    empty_rows['Open'] = selected_level
    empty_rows['Close'] = selected_level
    empty_rows['Low'] = selected_level
    empty_rows['High'] = selected_level

    result = pd.concat([df, empty_rows])
    return result


def stop_buy_order(ticker, open_price=1.0, stop_loss=0.9, take_profit=2.0,
                   comment='', lot_size=0.01):

    async def send_order():
        account, connection = await initialize_connection()

        print(open_price, type(open_price))

        try:
            result = await connection.create_stop_buy_order(
                symbol=ticker, volume=lot_size, open_price=open_price, stop_loss=stop_loss, take_profit=take_profit,
                options={'comment': 'comm', 'clientId': f'TE_{ticker}_7hyINWqAlE'}
            )
        except Exception as err:
            result = err

        return result

    return asyncio.run(send_order())


def stop_sell_order(ticker, open_price=1.0, stop_loss=0.9, take_profit=2.0,
                    comment='', lot_size=0.01):
    async def send_order():
        account, connection = await initialize_connection()

        print(open_price, type(open_price))

        try:
            result = await connection.create_stop_sell_order(
                symbol=ticker, volume=lot_size, open_price=open_price, stop_loss=stop_loss, take_profit=take_profit,
                options={'comment': 'comm', 'clientId': f'TE_{ticker}_7hyINWqAlE'}
            )
        except Exception as err:
            result = err

        return result

    return asyncio.run(send_order())


def get_horizontal_volumes(df):
    min_price = df['Low'].min()
    max_price = df['High'].max()
    delta = (max_price - min_price) / 100  # split all price track to 100 small levels

    horizontal_volumes = {k: 0 for k in range(0, 101)}

    for i, (index, row) in enumerate(df.iterrows()):
        candle_size = row['candle_size']

        start = int((row['Low'] - min_price) / delta)
        end = int((row['High'] - min_price) / delta)

        for k in range(start, end + 1):
            horizontal_volumes[k] += row['volume'] / (candle_size / delta)

    return horizontal_volumes
