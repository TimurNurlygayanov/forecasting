from uuid import uuid4
from crypto_forex.utils import ALL_TICKERS
from utils import get_data_alpha
from utils import draw


TICKERS = [t.split(':')[1] for t in sorted(ALL_TICKERS)]
delta = 0.00005

df = get_data_alpha(ticker='GBPUSD', interval='15min', limit=10000000)
for i, (index, row) in enumerate(df.iterrows()):

    if 20 < i < len(df) - 20:
        mean = sum(df['High'].values[i:i+4].tolist()) / 4
        diff = df['High'].values[i:i+4] - mean
        max_diff = abs(max(diff))

        mean = sum(df['Low'].values[i:i + 4].tolist()) / 4
        diff = df['Low'].values[i:i + 4] - mean
        min_diff = abs(max(diff))

        if row['High'] - row['Low'] > 3 * delta:
            if max_diff < delta or min_diff < delta:
                df_to_draw = df.iloc[i-10:i+14].copy()
                print(index)

                draw(df_to_draw, case_id='flat', custom_ticks=[mean, df_to_draw['High'].max(), df_to_draw['Low'].min()],
                     file_name=f'{uuid4()}', selected_level=mean)
