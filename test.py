import serial
import time

# Kết nối với Arduino
ser = serial.Serial('COM2', 9600, timeout=1)
time.sleep(2)  # Đợi Arduino khởi động

def send_command(cmd):
    ser.write((cmd + '\n').encode())
    time.sleep(0.5)  # Chờ Arduino thực hiện

while True:
    print("\nĐiều khiển hệ thống bắn bóng bàn:")
    print("1. Xoáy trái (spin_left)")
    print("2. Xoáy phải (spin_right)")
    print("3. Cấp bóng (feed)")
    print("4. Điều chỉnh góc bắn (servo:<góc>)")
    print("5. Dừng lại (stop)")
    print("6. Thoát")

    choice = input("Nhập lựa chọn: ")

    if choice == '1':
        send_command("spin_left")
    elif choice == '2':
        send_command("spin_right")
    elif choice == '3':
        send_command("feed")
    elif choice == '4':
        angle = input("Nhập góc servo (0-180): ")
        send_command(f"servo:{angle}")
    elif choice == '5':
        send_command("stop")
    elif choice == '6':
        print("Thoát chương trình.")
        break
    else:
        print("Lựa chọn không hợp lệ!")

ser.close()
