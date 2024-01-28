from crypto_forex.utils import ALL_TICKERS
from tqdm import tqdm
from bot.utils import get_data
from utils import get_data_alpha
from utils import draw
from utils import get_levels
from utils import get_horizontal_volumes

import plotly.graph_objects as go


TICKERS = [t.split(':')[1] for t in sorted(ALL_TICKERS)]

for ticker in tqdm(sorted(TICKERS)):
    # df = get_data(ticker, period='day', multiplier=1, days=100, save_data=False)
    # df_15_minutes = get_data(ticker, period='minute', multiplier=15, days=1, save_data=False)

    df = get_data_alpha(ticker, interval='Daily', limit=300)
    df_15_minutes = get_data_alpha(ticker, interval='15min', limit=100)

    print(df)

    df['candle_size'] = df['High'] - df['Low']
    average_atr = df['candle_size'].mean()
    price_diff = 0.02 * average_atr

    levels, _ = get_levels(df, price_diff=price_diff)

    draw(df_15_minutes, custom_ticks=levels, file_name=f'{ticker}', selected_level=0,
         case_id='forex_monitor', price_diff=price_diff*0.01, horizontal_volumes=None)

    with open(f'training_data/forex_monitor/{ticker}_deals.ini',
              encoding='utf8', mode='w+') as f:
        f.writelines('')

    exit(1)
