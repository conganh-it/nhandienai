# Hệ Thống Giám Sát Cảnh Báo Tự Động

## Mô Tả Dự Án
Dự án này phát triển một hệ thống giám sát an ninh sử dụng mô hình nhận diện đối tượng YOLOv5 kết hợp với một bot Telegram để gửi cảnh báo khi phát hiện người. Hệ thống sử dụng camera để thu thập hình ảnh và YOLOv5 để phát hiện sự hiện diện của con người. Nếu có người xuất hiện, bot Telegram sẽ gửi cảnh báo kèm theo ảnh chụp để người dùng có thể quyết định bật hoặc tắt âm thanh cảnh báo.

## Các Công Nghệ Sử Dụng

- **Python**: Ngôn ngữ lập trình chính sử dụng để xây dựng hệ thống.
- **YOLOv5**: Mô hình học sâu dùng để nhận diện người trong ảnh/video.
- **OpenCV**: Thư viện xử lý ảnh/video, được sử dụng để kết nối và xử lý video từ camera.
- **Aiogram**: Thư viện Python cho việc phát triển Telegram bot.
- **Pygame**: Thư viện dùng để phát âm thanh cảnh báo.
- **Torch**: Thư viện học sâu hỗ trợ việc sử dụng mô hình YOLOv5.

## Cấu Hình Hệ Thống

1. **Telegram Bot**: Bot Telegram được sử dụng để gửi cảnh báo và nhận lệnh bật/tắt cảnh báo từ người dùng. Bạn cần thay `TOKEN` và `CHAT_ID` bằng token của bot và ID của nhóm hoặc người dùng nhận cảnh báo.
2. **YOLOv5 Model**: Mô hình YOLOv5 đã được huấn luyện trước và có thể nhận diện người trong ảnh/video.
3. **Camera**: Camera (hoặc webcam) được sử dụng để quay video và cung cấp dữ liệu hình ảnh cho hệ thống.
4. **Pygame**: Dùng để phát âm thanh cảnh báo khi phát hiện người.

## Cách Cài Đặt

1. Cài đặt các thư viện cần thiết:
    ```bash
    pip install opencv-python torch aiogram pygame
    ```

2. Tải mô hình YOLOv5:
    Mô hình YOLOv5 được tải thông qua `torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)`.

3. Thay `TOKEN` và `CHAT_ID` trong mã nguồn:
    - `TOKEN`: Lấy từ [BotFather trên Telegram](https://core.telegram.org/bots#botfather).
    - `CHAT_ID`: ID của nhóm hoặc người dùng trên Telegram mà bạn muốn nhận cảnh báo.
 4. link yolo: https://miai.vn/?s=YOLO

## Các Lệnh Bot Telegram

- **/start**: Kích hoạt hệ thống giám sát.
- **/alert_on**: Bật cảnh báo âm thanh và gửi thông báo khi phát hiện người.
- **/alert_off**: Tắt cảnh báo âm thanh và ngừng gửi thông báo.

## Mô Tả Quy Trình Hoạt Động

1. **Giám sát liên tục**: Camera thu thập hình ảnh và gửi về hệ thống để phân tích.
2. **Nhận diện đối tượng**: YOLOv5 xử lý ảnh để phát hiện người.
3. **Cảnh báo và phản hồi từ người dùng**: 
    - Nếu phát hiện người và cảnh báo chưa bật, hệ thống sẽ gửi ảnh cho người dùng qua Telegram và yêu cầu bật/tắt cảnh báo.
    - Nếu cảnh báo đã bật, hệ thống sẽ phát âm thanh và gửi thông báo cảnh báo với ảnh kèm theo.

## Các Chức Năng Chính

- **Nhận diện người**: Sử dụng mô hình YOLOv5 để phát hiện sự hiện diện của con người trong video.
- **Cảnh báo qua Telegram**: Gửi thông báo và hình ảnh qua Telegram để người dùng có thể quyết định bật/tắt cảnh báo.
- **Âm thanh cảnh báo**: Sử dụng Pygame để phát âm thanh khi có người xuất hiện.

## Cấu Trúc Dự Án

```bash
.
├── camera_detection.py     # Quản lý việc nhận diện từ camera và gửi cảnh báo
├── telegram_bot.py         # Quản lý bot Telegram và giao tiếp với người dùng
├── requirements.txt        # Liệt kê các thư viện cần thiết
├── police.wav              # Tệp âm thanh cảnh báo
└── README.md               # File hướng dẫn dự án
