from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # Scratch
# model = YOLO("yolov8n.pt")  # official pre-trained
# model = YOLO("runs/detect/aeeg_yolov8n_50e8/weights/best.pt")  # check-point 1
# model = YOLO("runs/detect/aeeg_yolov8n_50e12/weights/best.pt")  # check-point 2
model = YOLO("runs/detect/aeeg_yolov8n_50e14/weights/best.pt")  # check-point 3

results = model.train(
   data='aeeg_epi_inter_v1.yaml',
   imgsz=1920,
   epochs=500,
   cache=True,
   batch=1,
   save=True,
   save_period=10,
   cos_lr=True,
   name='aeeg_yolov8n_50e',
   lr0=0.005,
   lrf=0.0025,
   hsv_h=0.1,
   hsv_s=0.1,
   hsv_v=0.1,
   translate=0,
   scale=0,
   fliplr=0,
   mosaic=0
)

# metrics = model.val(
#    data='aeeg_epi_inter_v1.yaml',
#    imgsz=1920,
#    batch=2,
#    name='aeeg_yolov8n_50e'
# )  # 在验证集上评估模型性能

metrics = model.val(
   task="detect",
   mode="val",
   model="runs/detect/aeeg_yolov8n_50e8/weights/best.pt",
   name="yolov8n_eval",
   data='aeeg_epi_inter_v1.yaml',
   imgsz=1920
)

# results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
