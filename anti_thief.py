import time
import cv2
import argparse
import numpy as np
from imutils.video import VideoStream
import imutils
import pyglet
import os

# Cài đặt tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--object_name', required=True, help='Tên vật thể cần phát hiện')
ap.add_argument('-f', '--frame', default=5, type=int, help='Số frame mất tích để báo động')
ap.add_argument('-c', '--config', default='yolov3.cfg', help='File config của YOLO')
ap.add_argument('-w', '--weights', default='yolov3.weights', help='File trọng số YOLO')
ap.add_argument('-cl', '--classes', default='yolov3.txt', help='File chứa danh sách vật thể')
args = ap.parse_args()

# Kiểm tra file tồn tại
for file in [args.config, args.weights, args.classes]:
    if not os.path.exists(file):
        print(f"Lỗi: Không tìm thấy {file}. Kiểm tra lại đường dẫn!")
        exit()

# Hàm lấy output layer của YOLO
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Hàm vẽ bounding box
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Khởi động camera
cap = VideoStream(src=0).start()
time.sleep(2.0)  # Chờ camera ổn định

# Đọc danh sách class từ file
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)

nCount = 0

# Vòng lặp đọc ảnh từ webcam
while True:
    frame = cap.read()
    image = imutils.resize(frame, width=600)

    Width, Height = image.shape[1], image.shape[0]
    scale = 0.00392

    # Tiền xử lý ảnh
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids, confidences, boxes = [], [], []
    conf_threshold, nms_threshold = 0.5, 0.4
    isExist = False

    # Xử lý đầu ra
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == args.object_name:
                center_x, center_y = int(detection[0] * Width), int(detection[1] * Height)
                w, h = int(detection[2] * Width), int(detection[3] * Height)
                x, y = center_x - w // 2, center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Áp dụng NMS để loại bỏ các bounding box trùng nhau
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if isinstance(indices, tuple):  # Kiểm tra nếu indices là tuple rỗng
        indices = []

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        if classes[class_ids[i]] == args.object_name:
            isExist = True
            draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))

    # Xử lý cảnh báo mất vật thể
    if isExist:
        nCount = 0
    else:
        nCount += 1
        if nCount > args.frame:
            cv2.putText(image, "ALARM! OBJECT MISSING!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            music = pyglet.resource.media('police.wav')
            music.play()

    cv2.imshow("Object Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
