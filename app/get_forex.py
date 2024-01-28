from crypto_forex.utils import ALL_TICKERS

from tqdm import tqdm


from utils import draw
from utils import get_levels
from utils import get_data_meta
from utils import get_atr
from utils import add_empty_rows
import os


TICKERS = [t.split(':')[1] for t in sorted(ALL_TICKERS)]
RESULTS = {}
profit_factor = 3
os.makedirs(f'training_data/forex_monitor')


for ticker in tqdm(TICKERS):
    df = get_data_meta(ticker, timeframe='1d', limit=100)

    levels, selected_level = get_levels(df)

    with open(f'app/levels.txt', 'r') as f:
        levels_from_file = [level.strip() for level in f.readlines()]
        manual_levels = {k.split(':')[0].strip(): float(k.split(':')[1].strip()) for k in levels_from_file}

    if not selected_level:
        selected_level = manual_levels.get(ticker, 0)

    if selected_level > 0:
        custom_ticks = [round(selected_level, 6),
                        df['High'].max(),
                        df['Low'].min(),
                        ]  # Add other default values as needed

        if ticker in manual_levels:
            custom_ticks.append(manual_levels[ticker])
            selected_level = manual_levels[ticker]

        df = add_empty_rows(df, length=3, selected_level=selected_level)

        atr = get_atr(df)
        tvh1 = selected_level + 0.02 * atr
        level_stop = tvh1 - 0.1 * atr
        take_profit1 = tvh1 + profit_factor * 0.1 * atr

        tvh2 = selected_level - 0.02 * atr
        level_stop2 = tvh2 + 0.1 * atr
        take_profit2 = tvh2 - profit_factor * 0.1 * atr

        draw(df, case_id='forex_monitor', custom_ticks=custom_ticks,
             file_name=f'{ticker}', selected_level=selected_level)

        # Get hourly timeframe -----

        df_hourly = get_data_meta(ticker, timeframe='1h', limit=70)
        df_hourly = add_empty_rows(df_hourly, length=3, selected_level=selected_level)
        custom_ticks = [round(selected_level, 6),
                        df_hourly['High'].max(),
                        df_hourly['Low'].min(),
                        df_hourly['High'].max() + 0.1*atr, df_hourly['Low'].min() - 0.1*atr]
        draw(df_hourly, case_id='forex_monitor', custom_ticks=custom_ticks,
             file_name=f'{ticker}_hourly', selected_level=selected_level)

        # Get 15m timeframe -----

        df_15_minutes_original = get_data_meta(ticker, timeframe='15m', limit=150)
        df_15_minutes = add_empty_rows(df_15_minutes_original.iloc[-70:], length=30, selected_level=selected_level)
        custom_ticks = [round(selected_level, 6),
                        df_15_minutes['High'].max(),
                        df_15_minutes['Low'].min(),
                        take_profit1, take_profit2]
        draw(df_15_minutes, case_id='forex_monitor', custom_ticks=custom_ticks,
             file_name=f'{ticker}_15minutes', selected_level=selected_level)

        # Detect levels on 15 minutes timeframe
        levels, selected_level_short = get_levels(df_15_minutes_original)

        df_15_minutes_original = add_empty_rows(df_15_minutes_original, length=30, selected_level=selected_level_short)

        custom_ticks = [round(selected_level_short, 6),
                        df_15_minutes['High'].max(),
                        df_15_minutes['Low'].min()]
        draw(df_15_minutes_original, case_id='forex_monitor', custom_ticks=custom_ticks,
             file_name=f'{ticker}_15minutes_levels', selected_level=selected_level_short)

        tvh1_short = selected_level_short + 0.02 * atr
        level_stop_short = tvh1_short - 0.1 * atr
        take_profit1_short = tvh1_short + profit_factor * 0.1 * atr

        tvh2_short = selected_level_short - 0.02 * atr
        level_stop2_short = tvh2_short + 0.1 * atr
        take_profit2_short = tvh2_short - profit_factor * 0.1 * atr

        ## ----

        deal = f"""[GLOBAL]
                atr={round(atr, 6)}
                level={round(selected_level, 6)}
                [DEAL1]
                tvh={round(tvh1, 6)}
                stop_loss={round(level_stop, 6)}
                take_profit={round(take_profit1, 6)}
                [DEAL2]
                tvh={round(tvh2, 6)}
                stop_loss={round(level_stop2, 6)}
                take_profit={round(take_profit2, 6)}
                [DEAL3]
                tvh={round(tvh1_short, 6)}
                stop_loss={round(level_stop_short, 6)}
                take_profit={round(take_profit1_short, 6)}
                [DEAL4]
                tvh={round(tvh2_short, 6)}
                stop_loss={round(level_stop2_short, 6)}
                take_profit={round(take_profit2_short, 6)}
                """
        with open(f'training_data/forex_monitor/{ticker}_deals.ini',
                  encoding='utf8', mode='w+') as f:
            f.writelines('\n'.join([line.strip() for line in deal.split()]))