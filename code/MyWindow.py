from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from serial_port_selector import SerialPortSelector
import serial
import cv2
import os
import sys
import datetime
import csv
from ultralytics import YOLO
import numpy as np
import threading
import platform

# 리소스 경로 함수 (PyInstaller 등 배포 시 리소스 경로 해결용)
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# UI 로드
try:
    ui_path = resource_path("pyqtapp3.ui")
    Form, Window = uic.loadUiType(ui_path)
except Exception as e:
    print(f"UI 파일 로드 실패: {e}")
    sys.exit()

class MyWindow(QMainWindow, Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 버튼 연결
        self.filmBut.clicked.connect(self.capture_photo)
        self.declarationBut.clicked.connect(self.declare_files)
        self.portBut.clicked.connect(self.open_port_selector)
        self.start_trainingBut.clicked.connect(self.start_training)

        # 상태 변수 초기화
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.model = YOLO(resource_path("runs/detect/train4/weights/best.pt"))
        self.detected_classes = []

        self.motion_ready = False
        self.selected_port = None
        self.training_thread = None

        self.image_file_name = None
        self.name = ""
        self.num = ""
        self.location = ""
        self.remark = ""

        self.start_camera()

    def start_camera(self):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0

        # 여러 장치번호 시도 (0,1,2)
        for device_id in range(3):
            self.cap = cv2.VideoCapture(device_id, backend)
            if self.cap.isOpened():
                print(f"카메라 장치 {device_id}를 열었습니다.")
                self.timer.start(30)
                return
            else:
                self.cap.release()
        QMessageBox.critical(self, "카메라 오류", "카메라를 열 수 없습니다. 장치 번호를 확인하세요.")
        sys.exit()

    def update_frame(self):
        if not (self.cap and self.cap.isOpened()):
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            print("프레임 읽기 실패, 재시도 중...")
            return

        results = self.model(frame)
        self.detected_classes = []

        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            self.detected_classes.append(f"{class_name} ({conf:.2f})")

            if class_name == "WildfireSmoke" and self.motion_ready:
                self.exeHumanoidMotion(19)

        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_Cam.setPixmap(QPixmap.fromImage(q_img))

    def capture_photo(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "경고", "카메라가 작동 중이지 않습니다.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_file_name = f"{timestamp}.png"
        self.lineEdit_file.setText(self.image_file_name)

        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame)
            annotated = results[0].plot()

            image_folder = resource_path("declaration/image")
            os.makedirs(image_folder, exist_ok=True)
            save_path = os.path.join(image_folder, self.image_file_name)
            cv2.imwrite(save_path, annotated)

            # 사용자 입력 값 저장
            self.name = self.lineEdit_Name.text().strip()
            self.num = self.lineEdit_Num.text().strip()
            self.location = self.lineEdit_location.text().strip()
            self.remark = self.textEdit_Remark.toPlainText().strip()

            label_folder = resource_path("declaration/labels")
            os.makedirs(label_folder, exist_ok=True)
            label_file = os.path.join(label_folder, f"{timestamp}.txt")
            with open(label_file, "w") as f:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cx, cy, w, h = box.xywh[0]
                    f.write(f"{cls_id} {cx} {cy} {w} {h}\n")

            QMessageBox.information(self, "촬영 완료", f"이미지 저장 완료: {save_path}")
        else:
            QMessageBox.critical(self, "오류", "사진 촬영 실패")

    def declare_files(self):
        if not self.image_file_name:
            QMessageBox.warning(self, "경고", "먼저 사진을 촬영하세요.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_info_path = resource_path("declaration/information")
        folder_path = os.path.join(base_info_path, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        original_image_path = os.path.join(resource_path("declaration/image"), self.image_file_name)
        target_image_path = os.path.join(folder_path, self.image_file_name)

        original_image = cv2.imread(original_image_path)
        if original_image is not None:
            cv2.imwrite(target_image_path, original_image)

        csv_name = f"{timestamp}.csv"
        csv_path = os.path.join(folder_path, csv_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Number", "Location", "Remark", "Image File", "Detected Classes"])
            writer.writerow([
                self.name,
                self.num,
                self.location,
                self.remark,
                self.image_file_name,
                ', '.join(self.detected_classes)
            ])

        QMessageBox.information(self, "신고 완료", f"정보가 저장되었습니다: {folder_path}")

    def exeHumanoidMotion(self, motion_id):
        if not self.motion_ready or not self.selected_port:
            return

        packet_buff = [0xff, 0xff, 0x4c, 0x53,
                       0x00, 0x00, 0x00, 0x00,
                       0x30, 0x0c, 0x03,
                       motion_id, 0x00, 0x64,
                       0x00]

        checksum = sum(packet_buff[6:14]) & 0xFF
        packet_buff[14] = checksum

        try:
            ser = serial.Serial(self.selected_port, 115200, timeout=1)
            if ser.is_open:
                ser.write(bytearray(packet_buff))
        except serial.SerialException as e:
            QMessageBox.warning(self, "시리얼 포트 오류", str(e))
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()

    def open_port_selector(self):
        selected_port = SerialPortSelector.launch(self)
        if selected_port:
            self.selected_port = selected_port
            self.portline.setText(selected_port)
            self.motion_ready = True

    def start_training(self):
        """현재 저장된 이미지 + 라벨로 YOLO 학습 시작"""
        self.start_trainingBut.setEnabled(False)

        image_folder = resource_path("declaration/image")
        label_folder = resource_path("declaration/labels")

        # 학습용 YAML 설정 (상대경로 기반으로 조절 가능)
        data_yaml = resource_path("data.yaml")
        with open(data_yaml, "w") as f:
            f.write(f"path: {os.path.dirname(image_folder)}\n")
            f.write("train: image\n")
            f.write("val: image\n")
            f.write("nc: 1\n")
            f.write("names: ['휴머노이드']\n")

        def run_training():
            model = YOLO("yolov8n.pt")  # 혹은 기존 best.pt로 미세조정 가능
            model.train(data=data_yaml, epochs=50, batch=16)
            # 학습 완료 후 버튼 다시 활성화 및 알림
            self.start_trainingBut.setEnabled(True)
            QMessageBox.information(self, "학습 완료", "모델 학습이 완료되었습니다.")

        self.training_thread = threading.Thread(target=run_training)
        self.training_thread.start()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

# 실행부
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
