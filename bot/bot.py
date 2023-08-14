import asyncio
from telegram import Bot

from tqdm import tqdm

from utils import get_data
from utils import get_tickets
from utils import check_strategy


with open('/Users/timur.nurlygaianov/telegram_token') as f:
    bot_token = ''.join(f.readlines()).strip()


RSI_THRESHOLD = 30
TO_BUY = {}
# Initialize the bot with your bot token
bot = Bot(token=bot_token)
chat_id = "335442091"


async def run_bot():

    TICKERS = get_tickets()

    await bot.send_message(chat_id=chat_id, text="Checking if there are some shares to buy...")

    for ticker in tqdm(TICKERS):
        print(ticker)
        data = get_data(ticker)

        if data is not None:
            # results, average_length = check_strategy(data, RSI_THRESHOLD)

            # Make sure the same strategy has more than 50% win rate in the past
            # for the same ticker, and make sure the price of 1 share is < $300
            #
            # if results > 0.5:
            if data['High'].values[-1] < 300:
                if data['Close'].values[-1] > data['EMA50'].values[-1] > data['EMA200'].values[-1]:
                    if data['EMA200'].values[-5] > data['EMA200'].values[-5]:
                        stop_loss = 0.92 * data['Close'].values[-1]
                        buy_price = data['Close'].values[-1]

                        msg = (f'🌴 buy *{ticker}* for ${buy_price:.2f} and set stop loss: ${stop_loss:.2f}')
                               # f'win rate on history data: {100*results:.1f}% within {average_length/10} days')

                        TO_BUY[ticker] = {'msg': msg, 'result': 0}

                        await bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')

    for res in sorted(TO_BUY.items(), key=lambda x: x[1]['result'], reverse=True):
        await bot.send_message(chat_id=chat_id, text=res[1]['msg'], parse_mode='Markdown')

    await bot.send_message(chat_id=chat_id, text="Finished!")


# Start the event loop
loop = asyncio.get_event_loop()

# Run the async function
loop.run_until_complete(run_bot())

# Close the event loop
loop.close()
