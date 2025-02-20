from ultralytics import YOLO
import cv2
import os

# 현재 작업 디렉토리 경로 확인
print("Current working directory: ", os.getcwd())

# 사전 학습된 yolov11n 모델 불러오기
model = YOLO("yolo11n.pt")

# vehicle 데이터셋으로 학습
print("Train YOLO model")
model.train(data='./datasets/vehicles/data.yaml', epochs=100, patience=10, batch=16, exist_ok=True, name='vehicles')

# validation 수행
print("Validate YOLO model")
model.val()

# 학습한 YOLO 모델 테스트
print("Predict YOLO model")
model_test = YOLO("./runs/detect/vehicles/weights/best.pt")
source = "./image_vehicle.jpg"
results = model_test(source)

# 테스트 결과 이미지 plot
res_plotted = results[0].plot() # 테스트 결과를 bb 그려서 plot 정보 저장장
cv2.imshow("test_result", res_plotted)
cv2.waitKey(0) # 아무 키 누를 때까지 대기
cv2.destroyAllWindows() # 모든 윈도우 창 닫기기
