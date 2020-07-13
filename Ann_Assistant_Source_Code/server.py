from bot import telegram_chatbot

bot = telegram_chatbot("config.cfg")

update_id = None
while True:
    updates = bot.get_updates(offset=update_id)
    updates = updates["result"]
    if updates:
        for item in updates:
            update_id = item["update_id"]
            try:
                message = str(item["message"]["text"])
                if message.lower()=="quit":
                    break
                rep = chat(message)
            except:
                message = None
                rep = None
            from_ = item["message"]["from"]["id"]
            bot.send_message(rep, from_)

