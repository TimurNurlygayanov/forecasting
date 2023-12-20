import pandas as pd
from flask import Flask
from flask import render_template
import glob
import base64
import configparser
import json
from plotly.utils import PlotlyJSONEncoder


config = configparser.ConfigParser()
app = Flask(__name__)


@app.route("/")
def index():
    results = []

    for filename in glob.iglob('training_data/*', recursive=False):
        results.append(filename.split('/')[1])

    return render_template('index.html', results=results)


@app.route("/case/<case_id>")
def case_page(case_id=''):
    if case_id:
        with open(f'training_data/{case_id}/5_minutes.png', "rb") as image_file:
            short_timeframe_before = base64.b64encode(image_file.read())
        with open(f'training_data/{case_id}/daily.png', "rb") as image_file:
            daily_timeframe_before = base64.b64encode(image_file.read())
        with open(f'training_data/{case_id}/5_minutes_after.png', "rb") as image_file:
            short_timeframe_after = base64.b64encode(image_file.read())

        short_timeframe_before = f'data:image/png;base64, {short_timeframe_before.decode()}'
        daily_timeframe_before = f'data:image/png;base64, {daily_timeframe_before.decode()}'
        short_timeframe_after = f'data:image/png;base64, {short_timeframe_after.decode()}'

        config.read(f'training_data/{case_id}/deal.ini')
        df = pd.read_csv(f'training_data/{case_id}/5_minutes_after.csv')
        results = [0, 0, 0, 0, 0]

        for i in range(1, 5):
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
            """} for i in range(1, 5)
        ]

        with open(f'training_data/{case_id}/5_minutes.json', 'r') as f:
            graph_5minutes = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')
        with open(f'training_data/{case_id}/5_minutes_after.json', 'r') as f:
            graph_5minutes_after = str(json.load(fp=f)).replace('False', 'false').replace('True', 'true')

        return render_template('case.html',
                               short_timeframe_before=short_timeframe_before,
                               daily_timeframe_before=daily_timeframe_before,
                               short_timeframe_after=short_timeframe_after,
                               deals=deals,
                               graph_5minutes=graph_5minutes,
                               graph_5minutes_after=graph_5minutes_after)



app.run()
