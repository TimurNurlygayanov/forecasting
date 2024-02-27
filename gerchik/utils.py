
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", message="urllib3")

import datetime

from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.graph_objs.layout.shape import Label
import plotly.graph_objects as go

import warnings

import numpy as np

from zigzag import peak_valley_pivots_detailed


warnings.filterwarnings('ignore', category=UserWarning)


def draw(df, file_name='', level=0, ticker='', boxes=None, future=0, second_levels=None,
         stop_loss=0, take_profit=0, buy_price=0, buy_index=0, zig_zag=False):
    start_moment = df.index[0]
    end_moment = df.index[-1]

    graph = make_subplots(rows=1, cols=1, shared_xaxes=False,
                          subplot_titles=[''])
    graph.update_layout(title="", xaxis_rangeslider_visible=False,
                        # xaxis=dict(showticklabels=False),
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

    # graph.update_xaxes(showticklabels=False, row=1, col=1)
    graph.update_xaxes(rangeslider={'visible': False}, row=1, col=1)
    graph.update_xaxes(type='category', categoryorder='trace')  # to ignore missed dates

    date_today = datetime.datetime.now().strftime("%b %d %Y")
    graph.update_layout(
        title={
            'text': f'{ticker}, daily timeframe<br>{date_today}',
            'y': 0.95,  # Adjust the vertical position of the title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)  # Adjust the font size
        }
    )

    if boxes:
        graph.update_layout(
            shapes=[
                dict(
                    type="rect",
                    x0=b['x0'],
                    y0=b['y0'],
                    x1=b['x1'],
                    y1=b['y1'],
                    line=dict(color=b['color'], width=2, dash="dash"),
                    fillcolor=b['color'],
                    label=Label({'text': b['label'], 'textposition': 'top center', 'yanchor': 'bottom'})
                ) for b in boxes
            ]
        )

    if level > 0:
        graph.add_shape(type='line', x0=start_moment, x1=end_moment, y0=level, y1=level,
                        line=dict(color='black', width=2),
                        row=1, col=1)

    if second_levels:
        for w in second_levels:
            graph.add_shape(type='line', x0=start_moment, x1=end_moment, y0=w, y1=w,
                            line=dict(color='rgba(10,10,134,0.7)', width=2),
                            row=1, col=1)

    """
    if future:
        graph.add_shape(type='rect', x0=len(df), x1=len(df)-future,
                        y0=min(df['Low'].tolist()[-future:]),
                        y1=max(df['High'].tolist()[-future:]),
                        line=dict(color='magenta', width=1),
                        row=1, col=1)
    """

    if stop_loss > 0 and take_profit > 0:
        graph.add_shape(type='rect', x0=buy_index, x1=len(df),
                        y0=buy_price,
                        y1=stop_loss,
                        fillcolor='rgba(200,10,10,0.2)',
                        line=dict(color='red', width=1),
                        row=1, col=1)
        graph.add_shape(type='rect', x0=buy_index, x1=len(df),
                        y0=buy_price,
                        y1=take_profit,
                        fillcolor='rgba(10,200,10,0.2)',
                        line=dict(color='green', width=1),
                        row=1, col=1)

    # df.ta.ema(length=50, append=True, col_names=('EMA50',))  # i use ema9 for stop loss on daily timeframe
    # scatter_trace = go.Scatter(x=df.index, y=df['EMA50'], mode='lines', name='EMA50')
    # graph.add_trace(scatter_trace)

    if zig_zag:
        X = np.array(df.iloc[-future:]['Close'].tolist())
        pivots = peak_valley_pivots_detailed(X, 0.4, -0.4, True, True)

        pivots_x = [df.index.tolist()[0]]
        pivots_y = [df['Close'].values[0]]
        for i, p in enumerate(pivots):
            color = 'rgba(10,200,14,0.5)'
            if p < 0:
                color = 'rgba(200,10,14,0.5)'

            if p != 0:
                y = df['Close'].values[i]
                x = df.index.tolist()[i]

                graph.add_scatter(x=[x],
                                  y=[y],
                                  marker=dict(
                                    color=color,
                                    size=20
                                  ),
                                  name='', showlegend=False)

                pivots_x.append(x)
                pivots_y.append(y)

        graph.add_trace(go.Scatter(x=pivots_x, y=pivots_y, name='ZigZag',
                                   line=dict(color='green', width=2, dash='dash'),
                                   showlegend=False))

    pio.write_image(
        graph, f'training_data/{file_name}.png',
        height=1000, width=2000
    )


def calculate_atr(df):
    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    candles_sizes = []
    for i in range(len(lows)):
        s = abs(highs[i] - lows[i])

        if s > 0:
            candles_sizes.append(s)

    # remove 10 % of smallest candles and 10 % of largest candles
    # so we are getting rid of enormous candles
    candles_sizes_medium = sorted(candles_sizes)[len(lows)//10:-len(lows)//10]
    medium_atr = sum(candles_sizes_medium) / len(candles_sizes_medium)

    # after we calculated medium atr it is time to sort candles one more time
    selected_candles = [s for s in candles_sizes if (medium_atr * 0.5) < s < (1.7 * medium_atr)]
    # now we are ready to provide true ATR:
    return sum(selected_candles[-5:]) / 5


def check_ratio(df_chunk, average_atr=1):
    days = len(df_chunk)

    if days < 4:
        return False, 1

    bottom = min(df_chunk['Low'].tolist())
    top = max(df_chunk['High'].tolist())
    # bottom = min(df_chunk['Open'].tolist() + df_chunk['Close'].tolist())
    # top = max(df_chunk['Open'].tolist() + df_chunk['Close'].tolist())

    # Exclude large bars from nakoplenie
    for i, (index, row) in enumerate(df_chunk.iterrows()):
        if row['High'] - row['Low'] > 1.5 * average_atr:
            return False, 1

    box_ratio = ((top - bottom) / average_atr) / days

    return box_ratio < 0.33, box_ratio


def find_nakoplenie(df, atr):
    best_ratio = 1
    RESULTS = []
    for size in range(4, len(df)):
        for start in range(0, len(df)-4):
            df_chunk = df.iloc[start:start + size].copy()
            result, ratio = check_ratio(df_chunk, average_atr=atr)

            if result and ratio < best_ratio:
                best_ratio = ratio

            if result:
                RESULTS.append({
                    'ratio': ratio, 'start': df_chunk.index[0], 'end': df_chunk.index[-1],
                    'start_int': start, 'end_int': start+size,
                    'min': min(df_chunk['Low']), 'max': max(df_chunk['High']),
                    'size': size
                })

    boxes = []
    RESULTS = sorted(RESULTS, key=lambda x: x['ratio'])
    drawn = []
    for r in RESULTS:
        status = True

        for d in drawn:
            if d['start_int'] <= r['start_int'] <= d['end_int']:
                status = False
            if d['start_int'] <= r['end_int'] <= d['end_int']:
                status = False

            if r['start_int'] <= d['start_int'] <= r['end_int']:
                status = False
            if r['start_int'] <= d['end_int'] <= r['end_int']:
                status = False

        if status and r['ratio'] < 0.2:
            drawn.append(r)

            boxes.append({
                'x0': r['start'], 'x1': r['end'], 'y0': r['max'], 'y1': r['min'],
                'start_int': r['start_int'], 'end_int': r['end_int'],
                'label': f"Накопление, ratio: {round(r['ratio'], 3)}, {r['size']} days",
                'color': 'rgba(255,155,34,0.1)'
            })

    return boxes


def check_for_bad_candles(df, atr):
    gaps = 0

    for i, (index, row) in enumerate(df.iterrows()):
        if i > 0:
            if abs(row['High'] - row['Low']) < 0.4 * atr:
                gaps += 1
            if abs(row['High'] - row['Low']) < 0.1:
                gaps += 1

    if gaps / len(df) > 0.05:
        return True

    return False


def search_for_bsu(lows, highs, bsu_price, luft):
    for i in range(len(lows) - 3):
        if abs(lows[i] - bsu_price) < 0.1 * luft:
            return True
        if abs(highs[i] - bsu_price) < 0.1 * luft:
            return True

    return False


def check_podzhatie(df):
    lows = df['Low'].tolist()
    highs = df['High'].tolist()
    opens = df['Open'].tolist()
    closes = df['Close'].tolist()
    s1 = highs[-1] - lows[-1]
    s2 = highs[-2] - lows[-2]
    s3 = highs[-3] - lows[-3]

    delta = 0.03

    if s1 < s2 < s3:   # volatilnost padaet
        if lows[-1] > lows[-2] > lows[-3]:
            if opens[-1] < closes[-1]:
                # Check for the confirmation
                k = 0
                for high in highs[-10:-1]:
                    if abs(high - highs[-1]) <= delta:
                        k += 1

                if k >= 2:
                    return highs[-1]  # draw(df, file_name=ticker, ticker=ticker, level=highs[-1])

        if highs[-1] < highs[-2] < highs[-3]:
            if opens[-1] > closes[-1]:
                k = 0
                for low in lows[-10:-1]:
                    if abs(low - lows[-1]) <= delta:
                        k += 1

                if k >= 2:
                    return lows[-1]  # draw(df, file_name=ticker, ticker=ticker, level=lows[-1])

    return 0


def check_scenario(df, level):
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    current_close = df['Close'].tolist()[-1]

    last_candle_size = highs[-1] - lows[-1]
    ratio = abs(level - current_close) / last_candle_size

    blizhnii_retest = False
    for i in range(4, 12):
        if lows[-i] < level < highs[-i]:
            blizhnii_retest = True

    label = f'Ratio: {round(ratio, 2):.2f}'
    if blizhnii_retest:
        label += '<br> Ближний ретест'
    else:
        label += '<br> Дальний ретест'

    return {
        'x0': df.index[-3], 'x1': df.index[-1], 'y0': min(lows[-3:]), 'y1': max(highs[-3:]),
        'label': label, 'color': 'rgba(55,200,34,0.2)'
    }


def check_simple_lp(df, level, atr, buy_price, stop_loss, take_profit):
    highs = df['High'].tolist()
    lows = df['Low'].tolist()
    current_open = df['Open'].tolist()[-1]
    previous_close = df['Close'].tolist()[-2]

    proshli_do_urovnia = level - previous_close

    label = f"Прошли {100 * proshli_do_urovnia / atr:.1f}% от ATR"
    top = 100 * (highs[-1] - level) / atr
    label += f"<br>Top {top:.1f}%"

    profit = 100 * abs(take_profit - buy_price) / buy_price
    loss = 100 * abs(stop_loss - buy_price) / buy_price

    label += f"<br>P: {profit:.1f}% L: {loss:.1f}%"

    color = 'rgba(200,10,10,0.01)'
    if top < 10:
        color = 'rgba(10,200,10,0.01)'

    return {
        'x0': df.index[0], 'x1': df.index[-1], 'y0': min(df['Low'].tolist()), 'y1': min(df['Low'].tolist()),
        'label': label, 'color': color
    }
