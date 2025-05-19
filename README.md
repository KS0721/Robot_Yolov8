# 🤖 Robot_Yolov8

## 기존에 만들어 둔 [🔗Yolo_pyqt](https://github.com/KS0721/Yolo_pyqt)를 활용해서 객체를 인식하면 로봇이 손을 흔드는 프로그램을 만들어 보았습니다.

### 1. 파일 생성 및 가상환경 만들기
#### 1) 기존에 만들어둔 Yolo_pyqt 파일의 복사본을 준비합니다. 
![image](https://github.com/user-attachments/assets/d2056f34-f855-468b-b697-f96fd94d4794)

저는 구분하기 쉽게 복사본의 폴더명을 변경했습니다.
#### 2) 새 가상환경을 만들고 해당 환경을 활성화합니다.
```
conda create -n pyQT_yolo  python=3.9 -y
conda activate pyQT_yolo
```
#### 3) YOLOv8 라이브러리 설치 
```
git clone https://github.com/ultralytics/ultralytics.git
pip install ultralytics
```
🚀 [Ultralytics YOLO 공식 GitHub 링크 바로가기](https://github.com/ultralytics/ultralytics)

#### 4) 필요한 패키지 설치
```
pip install -r requirements.txt
```
### ⚠️ COM 포트 인식 문제 해결 방법
USB 동글을 연결했는데도 COM 포트가 보이지 않거나 인식되지 않는 경우, 
[📥cp2104 driver](https://www.silabs.com/developer-tools/usb-to-uart-bridge-vcp-drivers?tab=downloads)를 설치해야 합니다..

<img src="https://github.com/user-attachments/assets/1ea3f6ba-b40f-4f24-af90-384bd909ae40" width="800" height="500">

## 2. 🧠 주요 코드 설명

#### 1) 객체 감지 및 자동 모션 트리거
```
if class_name == "휴머노이드" and self.motion_ready:
    self.exeHumanoidMotion(19)
```
#### 2) 시리얼 통신으로 모션 전송
```
packet_buff = [0xff, 0xff, 0x4c, 0x53, ...]
ser = serial.Serial(self.selected_port, 115200, timeout=1)
ser.write(bytearray(packet_buff))
```

## 🎬 시연 영상

### 휴머노이드를 감지했을 때, 로봇이 손을 흔드는 동작을 실행하는 모습입니다.

https://github.com/user-attachments/assets/1ab48dc2-5bf2-458f-a454-232b879aaec3

![image](https://github.com/user-attachments/assets/6602348e-c86b-4ab9-a6b7-abd296e9bad3)


### 📦 실행 파일 다운로드 
### [📥 Robot_Yolov8 다운로드](https://drive.google.com/file/d/1IHyVDWhgZI52Vx0wfrtnEuemE9UvhGPe/view?usp=sharing)



