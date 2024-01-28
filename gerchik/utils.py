import datetime

from plotly.subplots import make_subplots
import plotly.io as pio

from detecto.core import Dataset
from detecto.core import Model
from detecto.utils import read_image
from pathlib import Path
import warnings


warnings.filterwarnings('ignore', category=UserWarning)


def draw(df, file_name='', level=0, ticker=''):
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

    graph.add_shape(type='line', x0=start_moment, x1=end_moment, y0=level, y1=level,
                    line=dict(color='black', width=1),
                    row=1, col=1)

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
