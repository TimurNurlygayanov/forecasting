import datetime

from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.graph_objs.layout.shape import Label

from detecto.core import Dataset
from detecto.core import Model
from detecto.utils import read_image
from pathlib import Path
import warnings


warnings.filterwarnings('ignore', category=UserWarning)


def draw(df, file_name='', level=0, ticker='', boxes=None, future=0, second_levels=None,
         stop_loss=0, take_profit=0, buy_price=0, buy_index=0):
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
                            line=dict(color='rgba(10,10,134,0.2)', width=2),
                            row=1, col=1)

    if future:
        graph.add_shape(type='rect', x0=len(df), x1=len(df)-future,
                        y0=min(df['Low'].tolist()[-future:]),
                        y1=max(df['High'].tolist()[-future:]),
                        line=dict(color='magenta', width=1),
                        row=1, col=1)

    if stop_loss > 0 and take_profit > 0:
        graph.add_shape(type='rect', x0=buy_index, x1=len(df),
                        y0=buy_price,
                        y1=stop_loss,
                        line=dict(color='red', width=1),
                        row=1, col=1)
        graph.add_shape(type='rect', x0=buy_index, x1=len(df),
                        y0=buy_price,
                        y1=take_profit,
                        line=dict(color='green', width=1),
                        row=1, col=1)

    """
    df.ta.ema(length=21, append=True, col_names=('EMA21',))  # i use ema9 for stop loss on daily timeframe
    scatter_trace = go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21')
    graph.add_trace(scatter_trace)
    """

    pio.write_image(
        graph, f'training_data/{file_name}.png',
        height=1000, width=2000
    )


def detect(image_name, train=False):
    dataset = Dataset('/Users/timur.nurlygaianov/forecasting/marked_images')

    labels = ['p']

    model_name = 'saved_model.pth'
    if train:
        model = Model(labels)
        losses = model.fit(dataset)  # , epochs=15, learning_rate=0.001, verbose=True
        model.save(model_name)
    else:
        model = Model.load(model_name, labels)

    image = read_image(image_name)

    labels, boxes, scores = model.predict(image)
    return labels, boxes, scores


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
