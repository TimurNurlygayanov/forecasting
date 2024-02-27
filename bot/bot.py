from telegram import Bot



with open('/Users/timur.nurlygaianov/telegram_token') as f:
    bot_token = ''.join(f.readlines()).strip()


RSI_THRESHOLD = 30
TO_BUY = {}
# Initialize the bot with your bot token
bot = Bot(token=bot_token)
chat_id = "335442091"
