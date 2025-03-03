import os
import numpy as np
import cv2
import pyodbc
import json
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import pandas as pd
import onnxruntime

app = Flask(__name__)
app.secret_key = "secret"  # Secret key để quản lý session


# Kết nối SQL Server
def connect_to_db():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=CONGANH2711;"
        "DATABASE=ChamDiemDB;"
        "Trusted_Connection=yes;"
    )
    return conn


conn = connect_to_db()


# Hàm lấy danh sách điểm danh
def fetch_attendance_list(mssv=None):
    conn = connect_to_db()
    cursor = conn.cursor()

    if mssv:  # Nếu có MSSV => Chỉ lấy thông tin của sinh viên đó
        query = """
            SELECT in4_sv.MSSV, in4_sv.ho_ten, 
                   COALESCE(CONVERT(VARCHAR, Diem_danh.thoi_gian, 120), 'Chưa điểm danh') AS thoi_gian,
                   COALESCE(Diem_danh.trang_thai, 'Chưa điểm danh') AS trang_thai
            FROM in4_sv 
            LEFT JOIN Diem_danh ON in4_sv.MSSV = Diem_danh.MSSV
            WHERE in4_sv.MSSV = ?
        """
        cursor.execute(query, (mssv,))
    else:  # Nếu không có MSSV => Lấy toàn bộ danh sách
        query = """
            SELECT in4_sv.MSSV, in4_sv.ho_ten, 
                   COALESCE(CONVERT(VARCHAR, Diem_danh.thoi_gian, 120), 'Chưa điểm danh') AS thoi_gian,
                   COALESCE(Diem_danh.trang_thai, 'Chưa điểm danh') AS trang_thai
            FROM in4_sv 
            LEFT JOIN Diem_danh ON in4_sv.MSSV = Diem_danh.MSSV
            ORDER BY in4_sv.MSSV ASC
        """
        cursor.execute(query)

    result = cursor.fetchall()
    attendance_list = [{"MSSV": row[0], "ho_ten": row[1], "thoi_gian": row[2], "trang_thai": row[3]} for row in result]

    conn.close()
    return attendance_list


# Load model ONNX
onnx_session = onnxruntime.InferenceSession('C:/Users/8/PycharmProjects/chamdiem/final/face_recognition_model.onnx')

# Nhận diện khuôn mặt
detected_label = "Unknown"


def detect_face():
    global detected_label
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_label = "Unknown"
        for (x, y, w, h) in faces:
            detected_label = "Anh"  # Giả sử nhận diện được Anh
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute("SELECT MSSV FROM in4_sv WHERE ho_ten = ?", (detected_label,))
        result = cursor.fetchone()

        if result is None:
            return jsonify({'status': 'fail', 'message': 'Không tìm thấy MSSV cho sinh viên này'})

        mssv = result[0]
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state = "Đúng giờ"

        try:
            cursor.execute("INSERT INTO Diem_danh (MSSV, thoi_gian, trang_thai) VALUES (?, ?, ?)",
                           (mssv, time_now, state))
            conn.commit()

            return jsonify({'status': 'success', 'message': f'{detected_label} đã điểm danh - {state}'})
        except Exception as e:
            return jsonify({'status': 'fail', 'message': str(e)})

    return jsonify({'status': 'fail', 'message': 'Không nhận diện được khuôn mặt'})


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT loai FROM Login WHERE id=? AND pass=?", (username, password))
        user = cursor.fetchone()

        if user:
            session['username'] = username
            session['role'] = user[0]

            if user[0] == 'gv':
                return redirect(url_for('index_gv'))
            elif user[0] == 'sv':
                return redirect(url_for('index_sv'))

    return render_template('login.html')


@app.route('/index_sv')
def index_sv():
    if 'username' not in session or session['role'] != 'sv':
        return redirect(url_for('login'))

    attendance_list = fetch_attendance_list(session['username'])
    return render_template('index_sv.html', students=attendance_list)


@app.route('/index_gv')
def index_gv():
    if 'username' not in session or session['role'] != 'gv':
        return redirect(url_for('login'))

    attendance_list = fetch_attendance_list()
    return render_template('index_gv.html', students=attendance_list)



    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
