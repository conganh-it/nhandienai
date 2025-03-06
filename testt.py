import asyncio
import cv2
import torch
import time
import pygame
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile

# ============================
# 🔥 1️⃣ Cấu hình hệ thống
# ============================
TOKEN = "7977983640:AAGLMh26AWkTwJ3EdAbdOof29vny7Cy7jXI"  # 🔹 Thay bằng token của bot
CHAT_ID = "7052579864"  # 🔹 Thay bằng ID nhóm/người nhận cảnh báo

bot = Bot(token=TOKEN)
dp = Dispatcher()

# ============================
# 🔥 2️⃣ Load mô hình YOLOv5
# ============================
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# ============================
# 🔥 3️⃣ Cấu hình âm thanh cảnh báo
# ============================
pygame.mixer.init()
ALERT_SOUND = "police.wav"
pygame.mixer.music.load(ALERT_SOUND)

# ============================
# 🔥 4️⃣ Biến trạng thái
# ============================
alert_enabled = False  # Mặc định cảnh báo tắt, chỉ bật khi người dùng yêu cầu
last_alert_time = 0   # Thời gian gửi cảnh báo gần nhất
pending_alert = False  # Cờ kiểm tra đang chờ phản hồi từ người dùng

# ============================
# 🔥 5️⃣ Xử lý nhận diện
# ============================
def detect_person(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Kiểm tra có người xuất hiện không
    person_detected = any(detections['name'] == 'person')

    if person_detected:
        cv2.imwrite("alert.jpg", frame)  # Lưu ảnh cảnh báo
        return True
    return False

# ============================
# 🔥 6️⃣ Gửi ảnh phát hiện qua Telegram để người dùng quyết định
# ============================
async def send_detection(image_path):
    global pending_alert

    if not pending_alert:  # Nếu chưa có cảnh báo nào đang chờ xử lý
        pending_alert = True
        photo = FSInputFile(image_path)
        await bot.send_photo(
            chat_id=CHAT_ID,
            photo=photo,
            caption="👀 Phát hiện có người! Bạn có muốn bật cảnh báo không?\nGõ /alert_on để bật, /alert_off để tắt."
        )

# ============================
# 🔥 7️⃣ Gửi cảnh báo nếu người dùng đã bật
# ============================
async def send_alert():
    global last_alert_time

    if alert_enabled and time.time() - last_alert_time >= 30:
        last_alert_time = time.time()
        pygame.mixer.music.play(-1)  # 🔊 Bật âm thanh cảnh báo

        photo = FSInputFile("alert.jpg")
        await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption="🚨 CẢNH BÁO! Người lạ xuất hiện!")

# ============================
# 🔥 8️⃣ Điều khiển từ Telegram Bot
# ============================
@dp.message(Command("start"))
async def start_command(message: types.Message):
    await message.answer("🤖 Hệ thống giám sát đã kích hoạt!\nGõ /alert_on hoặc /alert_off để bật/tắt cảnh báo.")

@dp.message(Command("alert_on"))
async def alert_on(message: types.Message):
    global alert_enabled, pending_alert
    alert_enabled = True
    pending_alert = False  # Đã có quyết định từ người dùng
    await message.answer("🚨 Cảnh báo đã BẬT!")

@dp.message(Command("alert_off"))
async def alert_off(message: types.Message):
    global alert_enabled, pending_alert
    alert_enabled = False
    pending_alert = False  # Đã có quyết định từ người dùng
    pygame.mixer.music.stop()  # 🔇 Tắt âm thanh cảnh báo
    await message.answer("✅ Cảnh báo đã TẮT!")

# ============================
# 🔥 9️⃣ Chạy hệ thống giám sát
# ============================
async def camera_detection():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Vẽ giao diện dễ nhìn hơn
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(overlay, "Surveillance System", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if alert_enabled:
            cv2.putText(overlay, "ALERT: ON", (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "ALERT: OFF", (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Surveillance Camera", overlay)

        if detect_person(frame):
            if not alert_enabled:
                await send_detection("alert.jpg")  # Gửi ảnh cho người dùng kiểm tra trước
            await send_alert()  # Gửi cảnh báo nếu đã bật

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

# ============================
# 🔥 🔟 Chạy cả camera & Telegram song song
# ============================
async def main():
    cam_task = asyncio.create_task(camera_detection())
    telegram_task = dp.start_polling(bot)
    await asyncio.gather(cam_task, telegram_task)

if __name__ == "__main__":
    asyncio.run(main())
