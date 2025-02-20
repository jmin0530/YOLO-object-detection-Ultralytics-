# Ultralytics를 활용한 YOLO 모델 전이 학습

## Ultralytics 설치
```bash
$ pip install ultralytics
```

## 사전 학습된 YOLO 모델 불러오기
```python
model = YOLO("yolo11n.pt")
```

## YOLO 모델 전이 학습
```python
model.train(data='./datasets/vehicles/data.yaml', epochs=100, patience=10, batch=16, exist_ok=True, name='vehicles')
```
- 데이터셋은 Roboflow의 vehicle 데이터셋 사용 (https://universe.roboflow.com/roboflow-100/vehicles-q0x2v)

## 학습한 YOLO 모델 테스트
```python
model = YOLO("./runs/detect/vehicles/weights/best.pt")
source = "./image_vehicle.jpg"
results = model(source)
```

## 테스트 결과 이미지 plot
```python
res_plotted = results[0].plot() # 테스트 결과를 bb 그려서 plot 정보 저장
cv2.imshow("test_result", res_plotted)
cv2.waitKey(0) # 아무 키 누를 때까지 대기
cv2.destroyAllWindows() # 모든 윈도우 창 닫기기
```
