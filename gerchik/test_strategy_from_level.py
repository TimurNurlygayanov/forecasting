from bot.utils import get_data


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


R = 20
deal_length = 0
ticker = 'C:AUDCAD'  # Good deals 191, bad deals 311, result 1599
df = get_data(ticker, period='day', multiplier=1, save_data=True, days=500)
df_hourly = get_data(ticker, period='hour', multiplier=1, save_data=True, days=500)

# Last 100 days - is period we will check with trying to make deals

last_100_days = df[df.index < df_hourly.index.tolist()[-1000]]

hourly_lows = df_hourly['Low'].tolist()
hourly_highs = df_hourly['High'].tolist()
hourly_close = df_hourly['Close'].tolist()
hourly_open = df_hourly['Open'].tolist()


def search_for_bsu(df, bsu_price, luft):
    lows = df['Low'].tolist()
    highs = df['High'].tolist()

    for i in range(len(lows) - 1):
        if abs(lows[i] - bsu_price) < 0.1 * luft:
            return True
        if abs(highs[i] - bsu_price) < 0.1 * luft:
            return True

    return False


BAD_DEALS = 0
GOOD_DEALS = 0
for i in range(1000, 10, -1):
    k = -i
    daily_timeframe_before = df[df.index < df_hourly.index.tolist()[k]]

    current_atr = calculate_atr(daily_timeframe_before)
    stop_loss = 0.2 * current_atr
    luft = 0.1 * stop_loss
    order_price = 0
    stop_price = 0
    take_profit_price = 0

    found_signal = False
    if abs(hourly_lows[k] - hourly_lows[k+1]) < luft:
        if hourly_lows[k] <= hourly_lows[k+1]:
            if search_for_bsu(daily_timeframe_before, bsu_price=hourly_lows[k], luft=luft):
                found_signal = True
                order_price = hourly_lows[k] + luft
                stop_price = order_price - stop_loss
                take_profit_price = order_price + R * stop_loss

    if abs(hourly_highs[k] - hourly_highs[k+1]) < luft:
        if hourly_highs[k] >= hourly_highs[k+1]:
            if search_for_bsu(daily_timeframe_before, bsu_price=hourly_highs[k], luft=luft):
                found_signal = True
                order_price = hourly_highs[k] - luft
                stop_price = order_price + stop_loss
                take_profit_price = order_price - R * stop_loss

    if found_signal:
        hourly_after_the_deal = df_hourly[df_hourly.index > df_hourly.index.tolist()[k]]

        deal_started = False
        deal_finished = False
        for w, (index, row) in enumerate(hourly_after_the_deal.iterrows()):

            if not deal_started and w <= 0 and row['Low'] < order_price < row['High']:
                deal_started = True

            if deal_started and not deal_finished:
                if stop_price < take_profit_price:  # Long
                    if row['Low'] < stop_price:
                        BAD_DEALS += 1
                        deal_finished = True
                        deal_length += w + 1
                    if row['High'] > take_profit_price:
                        GOOD_DEALS += 1
                        deal_finished = True
                        deal_length += w + 1
                else:
                    # Short:
                    if row['High'] > stop_price:
                        BAD_DEALS += 1
                        deal_finished = True
                        deal_length += w + 1
                    if row['Low'] < take_profit_price:
                        GOOD_DEALS += 1
                        deal_finished = True
                        deal_length += w + 1

print(f"Good deals {GOOD_DEALS}, bad deals {BAD_DEALS}, result {-BAD_DEALS + R*GOOD_DEALS}, length: {deal_length / (GOOD_DEALS + BAD_DEALS):.1f}")
