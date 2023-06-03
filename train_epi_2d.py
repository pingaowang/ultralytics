from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # Scratch
model = YOLO("yolov8n.pt")  # official pre-trained
# model = YOLO("runs/detect/aeeg_yolov8n_50e8/weights/best.pt")  # check-point 1
# model = YOLO("runs/detect/aeeg_yolov8n_50e12/weights/best.pt")  # check-point 2
# model = YOLO("runs/detect/aeeg_yolov8n_50e14/weights/best.pt")  # check-point 3
# model = YOLO("runs/detect/aeeg_yolov8n_50ev2/weights/best.pt")  # check-point 4
# model = YOLO("runs/detect/aeeg_yolov8n_50ev3/weights/best.pt")  # check-point 5
# model = YOLO("runs/detect/aeeg_yolov8n_50ev4_only_epi_/weights/best.pt")  # check-point 6
# model = YOLO("runs/detect/aeeg_yolov8n_50ev4_only_epi_5/weights/best.pt")  # check-point inter-1
# model = YOLO("runs/detect/aeeg_yolov8n_OnlyInter_Adam_lr1e-3_wd1e-4/weights/best.pt")  # check-point inter-2
# model = YOLO("runs/detect/aeeg_yolov8n_OnlyInter_Adam_lr1e-3_wd1e-42/weights/best.pt")  # check-point inter-3
# model = YOLO("runs/detect/aeeg_yolov8n_OnlyInter_Adam_lr1e-3_wd1e-43/weights/best.pt")  # check-point inter-4
# model = YOLO("runs/detect/aeeg_2d_onlyEpi/weights/best.pt")  # check-point 2d only-epi 1
model = YOLO("runs/detect/aeeg_2d_onlyEpi3/weights/best.pt")  # check-point 2d only-epi 2

results = model.train(
   data='aeeg_epi_2d_v1.yaml',
   agnostic_nms=True,
   single_cls=True,
   imgsz=1800,
   epochs=300,
   patience=1000,
   optimizer="Adam",
   cache=True,
   batch=2,
   cos_lr=True,
   name='aeeg_2d_onlyEpi',
   lr0=1.0e-03,
   lrf=0.2,
   weight_decay=1.0e-04,
   warmup_epochs=0.0,
   hsv_h=0.1,
   hsv_s=0.1,
   hsv_v=0.1,
   # translate=0,
   # scale=0,
   fliplr=0.5,
   flipud=0.5,
   mosaic=1,
   mixup=0.2,
   # perspective=0.1,
   degrees=5
   # dropout=0.9
)

# metrics = model.val(
#    data='aeeg_epi_inter_v1.yaml',
#    imgsz=1920,
#    batch=2,
#    name='aeeg_yolov8n_50e'
# )  # 在验证集上评估模型性能

# metrics = model.val(
#    task="detect",
#    mode="val",
#    model="runs/detect/aeeg_yolov8n_50e8/weights/best.pt",
#    name="yolov8n_eval",
#    data='aeeg_epi_inter_v1.yaml',
#    imgsz=1920
# )

# results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
# success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
