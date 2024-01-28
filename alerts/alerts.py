import asyncio
from telegram import Bot

from datetime import datetime

from tickers import ALL_TICKERS
from utils import get_data_alpha


with open('/Users/timur.nurlygaianov/telegram_token') as f:
    bot_token = ''.join(f.readlines()).strip()

chat_id = "335442091"
TICKERS = [t.split(':')[1] for t in sorted(ALL_TICKERS)]


def send_message(message_text):
    bot = Bot(token=bot_token)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.send_message(chat_id=chat_id, text=message_text))


def run_bot():
    msg = '= ' * 30
    msg += '\n' + datetime.now().strftime("%d %B, %Y  %H:%M:%S")
    print(msg)

    def run_me(ticker):
        with open('/Users/timur.nurlygaianov/forecasting/alerts/levels.txt', mode='r', encoding='utf-8') as f:
            levels = f.readlines()

        levels = [level.strip() for level in levels if level.strip() > "" and ticker in level]

        if levels:
            levels = levels[0].split()[1:]

            df = get_data_alpha(ticker, interval='5min', limit=10)

            alert_triggered = False
            for level in levels:
                for i, (index, row) in enumerate(df.iterrows()):
                    if not alert_triggered:
                        if row['Low'] < float(level) < row['High']:
                            send_message(msg + f"\n\n {ticker} price level {level} alert!")
                            alert_triggered = True

    for ticker in TICKERS:
        run_me(ticker)


if __name__ == "__main__":
    non_working_days = ['Saturday', 'Sunday']
    day_of_the_week = datetime.now().strftime('%A')

    exit(0)

    if day_of_the_week not in non_working_days:
        run_bot()
