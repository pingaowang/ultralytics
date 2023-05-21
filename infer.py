import os

from ultralytics import YOLO

# model = YOLO("runs/detect/aeeg_yolov8n_50e8/weights/best.pt")
model = YOLO("runs/detect/aeeg_yolov8n_50e12/weights/best.pt")

test_image_paths = [os.path.join("dataset_yolo/images/test", x) for x in os.listdir("dataset_yolo/images/test")]
for image_path in test_image_paths:
    results = model.predict(image_path)  # 对图像进行预测

    print(results[0].boxes.data)

print(0)