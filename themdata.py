import sqlite3
import os

DB_PATH = os.path.join("final/db_mysql", "Nhan_dien.db")

# Kết nối đến SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1️⃣ Chèn dữ liệu vào bảng Login
cursor.executemany("""
INSERT INTO Login (id, pass, loai) VALUES (?, ?, ?)
""", [
    ("admin", "admin123", "gv"),
    ("k205480106001", "123456", "sv"),
    ("k205480106002", "123456", "sv"),
    ("k205480106003", "123456", "sv")
])

# 2️⃣ Chèn dữ liệu vào bảng in4_sv
cursor.executemany("""
INSERT INTO in4_sv (MSSV, ho_ten, lop, khoa) VALUES (?, ?, ?, ?)
""", [
    ("K205480106001", "anh", "K56KMT", "Kỹ Thuật Máy Tính"),
    ("K205480106002", "huy", "K56KMT", "Kỹ Thuật Máy Tính"),
    ("K205480106003", "hong", "K56KMT", "Kỹ Thuật Máy Tính")
])

# 3️⃣ Chèn dữ liệu vào bảng Thoigian_tiet
cursor.execute("""
INSERT INTO Thoigian_tiet (gio, phut) VALUES (?, ?)
""", (7, 30))  # Tiết bắt đầu từ 7:30

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("✅ Dữ liệu mẫu đã được thêm vào database!")
