import pandas as pd
from flask import Flask
from flask import request
from flask import render_template
import glob
import base64
import configparser
import json
from plotly.utils import PlotlyJSONEncoder
from crypto_forex.utils import ALL_TICKERS

from bot.utils import get_data
from utils import get_levels
from utils import draw
from utils import get_data_meta
from utils import stop_buy_order
from utils import stop_sell_order


config = configparser.ConfigParser()
app = Flask(__name__)


@app.route("/")
def index():
    results = []

    for filename in glob.iglob('training_data/*', recursive=False):
        if 'forex' not in filename:
            results.append(filename.split('/')[1])

    return render_template('index.html', results=results)


@app.route("/case/<case_id>")
def case_page(case_id=''):
    if case_id:
        config.read(f'training_data/{case_id}/deal.ini')
        df = pd.read_csv(f'training_data/{case_id}/5_minutes_after.csv')
        results = [0, 0, 0, 0, 0, 0, 0]

        for i in range(1, 7):
            tvh = float(config[f"DEAL{i}"]["tvh"])
            stop_loss = float(config[f"DEAL{i}"]["stop_loss"])
            take_profit = float(config[f"DEAL{i}"]["take_profit"])

            deal_active = False
            result = 0

            for k in range(0, 30):
                row = df.iloc[k-30]

                if not deal_active and result == 0:
                    if row['Low'] < tvh < row['High']:
                        deal_active = True

                if deal_active:
                    if row['Low'] < stop_loss < row['High']:
                        result = -1
                        deal_active = False
                    elif row['Low'] < take_profit < row['High']:
                        result = 1
                        deal_active = False

            results[i] = result

        deals = [{
            'result': results[i],
            'tvh': config[f"DEAL{i}"]["tvh"],
            'take_profit': config[f"DEAL{i}"]["take_profit"],
            'stop_loss': config[f"DEAL{i}"]["stop_loss"],
            'html': f"""<p><b>TVH</b>: {config[f"DEAL{i}"]["tvh"]} </p>
            <p><b>Stop Loss</b>: {config[f"DEAL{i}"]["stop_loss"]} </p>
            <p><b>Take Profit</b>: {config[f"DEAL{i}"]["take_profit"]} </p>
            """} for i in range(1, 7)
        ]

        with open(f'training_data/{case_id}/daily.json', 'r') as f:
            graph_daily = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/{case_id}/hourly.json', 'r') as f:
            graph_hourly = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/{case_id}/5_minutes.json', 'r') as f:
            graph_5minutes = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/{case_id}/5_minutes_after.json', 'r') as f:
            graph_5minutes_after = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')

        return render_template('case.html',
                               graph_daily=graph_daily,
                               graph_hourly=graph_hourly,
                               deals=deals,
                               graph_5minutes=graph_5minutes,
                               graph_5minutes_after=graph_5minutes_after)


@app.route("/forex")
def forex_tickers():
    graphs = []
    for filename in glob.iglob('training_data/forex_monitor/*.ini', recursive=False):
        ticker = filename.split('/')[-1].split('_')[0].split('.')[0]

        with open(f'training_data/forex_monitor/{ticker}.png', "rb") as image_file:
            graph = base64.b64encode(image_file.read())
            image = f'data:image/png;base64, {graph.decode()}'

            graphs.append({'ticker': ticker, 'graph': image, 'color': '#FFFFCC'})

        """
        with open(f'training_data/forex_monitor/{ticker}_15minutes_levels.png', "rb") as image_file:
            graph = base64.b64encode(image_file.read())
            image = f'data:image/png;base64, {graph.decode()}'

            graphs.append({'ticker': ticker, 'graph': image, 'color': '#00FF99'})
        """

    return render_template('forex_table.html',
                           graphs=graphs)


@app.route("/forex_latest_data/<ticker>")
def forex_ticker_new_data(ticker):
    formatted_ticker = f'C:{ticker}'
    stop_loss_money = 200  # $200
    stop_loss_pips = 20  # sometimes it can be 10 pips, but in most cases 20 is better
    risk_reward_ratio = 3

    if formatted_ticker in ALL_TICKERS:
        df = get_data_meta(ticker, timeframe='1d', limit=100)

        with open(f'app/levels.txt', 'r') as f:
            levels_from_file = [level.strip() for level in f.readlines()]
            manual_levels = {k.split(':')[0].strip(): float(k.split(':')[1].strip()) for k in levels_from_file}

        custom_ticks, selected_level = get_levels(df)

        if ticker in manual_levels:
            custom_ticks.append(manual_levels[ticker])
            selected_level = manual_levels[ticker]

        draw(df, case_id=ticker, custom_ticks=custom_ticks,
             file_name=f'{ticker}', selected_level=selected_level)


@app.route("/forex/<ticker>")
def forex_ticker_data(ticker):
    formatted_ticker = f'C:{ticker}'
    if formatted_ticker in ALL_TICKERS:
        with open(f'training_data/forex_monitor/{ticker}.json', 'r') as f:
            graph_daily = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/forex_monitor/{ticker}_hourly.json', 'r') as f:
            graph_hourly = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/forex_monitor/{ticker}_15minutes.json', 'r') as f:
            graph_15minutes = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/forex_monitor/{ticker}_15minutes_levels.json', 'r') as f:
            graph_15minutes_levels = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')

        config.read(f'training_data/forex_monitor/{ticker}_deals.ini')

        deals = [{
            'tvh': config[f"DEAL{i}"]["tvh"],
            'take_profit': config[f"DEAL{i}"]["take_profit"],
            'stop_loss': config[f"DEAL{i}"]["stop_loss"],
            'html': f"""<p><b>TVH</b>: {config[f"DEAL{i}"]["tvh"]} </p>
                    <p><b>Stop Loss</b>: {config[f"DEAL{i}"]["stop_loss"]} </p>
                    <p><b>Take Profit</b>: {config[f"DEAL{i}"]["take_profit"]} </p>
                    """
            } for i in range(1, 5)
        ]

        atr = round(float(config[f"GLOBAL"]["atr"]), 7)
        stop_loss_size = round(float(atr) * 0.2, 7)

        return render_template(
            'case_forex_deal.html',
            atr=config[f"GLOBAL"]["atr"],
            stop_loss_size=stop_loss_size,
            graph_daily=graph_daily,
            graph_15minutes=graph_15minutes,
            graph_hourly=graph_hourly,
            deals=deals,
            ticker=ticker,
            graph_15minutes_levels=graph_15minutes_levels
        )


@app.post("/send_order")
def send_order():
    data = request.get_json()
    ticker = data.get('ticker')
    open_price = data.get('open_price')
    stop_loss = data.get('stop_loss')
    take_profit = data.get('take_profit')
    comment = data.get('comment')
    lot_size = float(data.get('lot_size', 0.01))

    result = ''

    if take_profit < stop_loss:
        print('SELL', ticker, open_price, stop_loss, take_profit)
        result = stop_sell_order(ticker=ticker, open_price=open_price,
                                 stop_loss=stop_loss, take_profit=take_profit, comment=comment,
                                 lot_size=lot_size)
    else:
        print('BUY', ticker, open_price, stop_loss, take_profit)
        result = stop_buy_order(ticker=ticker, open_price=open_price, stop_loss=stop_loss,
                                take_profit=take_profit, comment=comment,
                                lot_size=lot_size)

    print('RESULT', result)

    return result


app.run()
