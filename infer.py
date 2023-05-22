import os
import pickle
from ultralytics import YOLO


model_name = "aeeg_yolov8n_50e16"

model = YOLO("runs/detect/{}/weights/best.pt".format(model_name))
result_save_root = "prediction_results"
save_folder = os.path.join(result_save_root, model_name)
os.makedirs(save_folder, exist_ok=True)

test_image_paths = [os.path.join("dataset_yolo/images/test", x) for x in os.listdir("dataset_yolo/images/test")]
for image_path in test_image_paths:
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_folder, image_name.split('.')[0] + ".pkl")

    results = model.predict(image_path)  # 对图像进行预测

    bboxes_tensor = results[0].boxes.data
    print("==== {}: # bboxes: {} ====".format(image_name, bboxes_tensor.size()[0]))

    bboxes_to_save = []
    for i_bbox in range(bboxes_tensor.size()[0]):
        pred_bbox = bboxes_tensor[i_bbox, :]

        cls = pred_bbox.int()[5].item()
        xmin = pred_bbox.int()[0].item()
        ymin = pred_bbox.int()[1].item()
        xmax = pred_bbox.int()[2].item()
        ymax = pred_bbox.int()[3].item()
        conf = pred_bbox[4].item()

        bboxes_to_save.append([cls, xmin, ymin, xmax, ymax, conf])

    with open(save_path, 'wb') as f:
        pickle.dump(bboxes_to_save, f)
