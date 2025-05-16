# 🤖 Robot_Yolov8

## 기존에 만들어 둔 [🔗Yolo_pyqt](https://github.com/KS0721/Yolo_pyqt)를 활용해서 객체를 인식하면 로봇이 손을 흔드는 프로그램을 만들어 보았습니다.

### 1. 파일 생성 및 가상환경 만들기
#### 1) 기존에 만들어둔 Yolo_pyqt 파일의 복사본을 하나 더 준비합니다. 
![image](https://github.com/user-attachments/assets/d2056f34-f855-468b-b697-f96fd94d4794)

저는 구분하기 쉽게 파일명을 바꿨습니다.
#### 2) 새 가상환경을 만들고 해당 환경을 활성화합니다.
```
conda create -n pyQT_yolo  python=3.9 -y
conda activate pyQT_yolo
```
#### 3) YOLOv8 코드 가져오기 
```
git clone https://github.com/ultralytics/ultralytics.git
pip install ultralytics
```
🚀 [Ultralytics YOLO 공식 GitHub 링크 바로가기](https://github.com/ultralytics/ultralytics)

#### 4) 다운로드한 라이브러리 설치
```
pip install -r requirements.txt
```
### ⚠️ COM 포트 인식 문제 해결 방법
