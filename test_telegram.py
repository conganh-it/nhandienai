import asyncio
import telegram

my_token = "7977983640:AAGLMh26AWkTwJ3EdAbdOof29vny7Cy7jXI"
#TOKEN = "7977983640:AAGLMh26AWkTwJ3EdAbdOof29vny7Cy7jXI"
#CHAT_ID = "7052579864"


async def send_message():
    bot = telegram.Bot(token=my_token)
    await bot.send_message(chat_id="7052579864", text="Gửi từ PyCharm")


# Kiểm tra nếu event loop đã chạy, thì dùng nó luôn
def run_async_function(async_func):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(async_func)


# Chạy gửi tin nhắn
run_async_function(send_message())
