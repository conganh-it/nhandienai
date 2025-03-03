import os
import numpy as np
import cv2
import sqlite3
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from scipy.spatial.distance import cosine
import pandas as pd
import onnxruntime

app = Flask(__name__)

# Kết nối CSDL SQLite
def connect_to_db():
    return sqlite3.connect('C:/Users/8/PycharmProjects/chamdiem/final/db_mysql/Nhan_dien.db', check_same_thread=False)

conn = connect_to_db()

# Lấy danh sách sinh viên và trạng thái điểm danh
def fetch_attendance_list():
    query = """
        SELECT in4_sv.MSSV, in4_sv.ho_ten, 
               COALESCE(Diem_danh.thoi_gian, 'Chưa điểm danh') as thoi_gian, 
               COALESCE(Diem_danh.trang_thai, 'Chưa điểm danh') as trang_thai
        FROM in4_sv 
        LEFT JOIN Diem_danh ON in4_sv.MSSV = Diem_danh.MSSV
        ORDER BY in4_sv.MSSV ASC
    """
    df = pd.read_sql(query, conn)
    return df.to_dict(orient='records')

# Nhận diện khuôn mặt
onnx_session = onnxruntime.InferenceSession('C:/Users/8/PycharmProjects/chamdiem/final/face_recognition_model.onnx')
dataset_dir = 'C:/Users/8/PycharmProjects/chamdiem/final/dataset'
dataset_embeddings = []
dataset_labels = []

for label in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        face = cv2.imread(image_path)
        if face is not None:
            face = cv2.resize(face, (160, 160)).astype('float32')
            face = (face / 255.0 - 0.5) / 0.5
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, axis=0)

            input_name = onnx_session.get_inputs()[0].name
            face_embedding = onnx_session.run(None, {input_name: face})[0].flatten()
            dataset_embeddings.append(face_embedding)
            dataset_labels.append(label)

dataset_embeddings = np.array(dataset_embeddings)
dataset_labels = np.array(dataset_labels)

detected_label = None

def detect_face():
    global detected_label
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_label = None
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160)).astype('float32')
            face = (face / 255.0 - 0.5) / 0.5
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, axis=0)

            input_name = onnx_session.get_inputs()[0].name
            face_embedding = onnx_session.run(None, {input_name: face})[0].flatten()

            distances = [cosine(face_embedding, stored_embedding) for stored_embedding in dataset_embeddings]
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            label = dataset_labels[min_distance_idx] if min_distance < 0.6 else "Unknown"
            detected_label = label

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(detect_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/diemdanh', methods=['POST'])
def diemdanh():
    global detected_label
    if detected_label and detected_label != "Unknown":
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state = "Đúng giờ"

        query = "INSERT INTO Diem_danh (MSSV, thoi_gian, trang_thai) VALUES (?, ?, ?)"
        cursor = conn.cursor()
        cursor.execute(query, (detected_label, time_now, state))
        conn.commit()

        attendance_list = fetch_attendance_list()
        return jsonify({'status': 'success', 'message': f'{detected_label} đã điểm danh - {state}', 'attendance': attendance_list})

    return jsonify({'status': 'fail', 'message': 'Không nhận diện được khuôn mặt'})

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/index_sv')
def index_sv():
    attendance_list = fetch_attendance_list()
    return render_template('index_sv.html', students=attendance_list)

@app.route('/index_gv')
def index_gv():
    return render_template('index_gv.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data.get('id')
    password = data.get('pass')

    cursor = conn.cursor()
    query = "SELECT loai FROM Login WHERE id = ? AND pass = ?"
    cursor.execute(query, (user_id, password))
    user = cursor.fetchone()

    if user:
        loai = user[0]
        return jsonify({'status': 'success', 'redirect': url_for('index_sv' if loai == 'sv' else 'index_gv')})
    else:
        return jsonify({'status': 'fail', 'message': 'Sai ID hoặc mật khẩu'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
