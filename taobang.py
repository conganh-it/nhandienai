import sqlite3
import os

# Định nghĩa đường dẫn đến thư mục db_mysql
DB_FOLDER = "db_mysql"
DB_PATH = os.path.join(DB_FOLDER, "Nhan_dien.db")

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# Kết nối với SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1️⃣ Tạo bảng Login - Lưu thông tin tài khoản
cursor.execute("""
CREATE TABLE IF NOT EXISTS Login (
    id TEXT PRIMARY KEY,
    pass TEXT NOT NULL,
    loai TEXT NOT NULL  -- 'gv' (giáo viên) hoặc 'sv' (sinh viên)
);
""")

# 2️⃣ Tạo bảng in4_sv - Lưu thông tin sinh viên
cursor.execute("""
CREATE TABLE IF NOT EXISTS in4_sv (
    MSSV TEXT PRIMARY KEY,
    ho_ten TEXT NOT NULL,
    lop TEXT NOT NULL,
    khoa TEXT NOT NULL
);
""")

# 3️⃣ Tạo bảng Thoigian_tiet - Lưu thời gian tiết học
cursor.execute("""
CREATE TABLE IF NOT EXISTS Thoigian_tiet (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gio INTEGER NOT NULL,
    phut INTEGER NOT NULL
);
""")

# 4️⃣ Tạo bảng Diem_danh - Lưu thông tin điểm danh
cursor.execute("""
CREATE TABLE IF NOT EXISTS Diem_danh (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MSSV TEXT NOT NULL,
    thoi_gian TEXT NOT NULL,
    trang_thai TEXT NOT NULL,
    FOREIGN KEY (MSSV) REFERENCES in4_sv (MSSV)
);
""")

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("✅ Database và các bảng đã được tạo thành công!")
