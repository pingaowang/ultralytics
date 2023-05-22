import os
from PIL import Image, ImageDraw, ImageFont
import pickle
from utils.load_coco_form import load_coco_annotations_as_coords

# ! Change here, for a different model's predictions.
prediction_pkl_folder = "prediction_results/aeeg_yolov8n_50e16"

# root info, never change.
yolo_dataset_root = "/home/pingao/projects/datasets/dataset_yolo"
vis_save_root = "pred_and_label_vis"

# makedir vis image saving folder
prediction_pkl_folder_name = os.path.basename(prediction_pkl_folder)
vis_save_folder = os.path.join(vis_save_root, prediction_pkl_folder_name)
os.makedirs(vis_save_folder, exist_ok=True)

# get input paths
yolo_test_image_folder = os.path.join(yolo_dataset_root, "images", "test")
yolo_test_label_folder = os.path.join(yolo_dataset_root, "labels", "test")
test_image_names = os.listdir(yolo_test_image_folder)
test_image_paths = [os.path.join(yolo_test_image_folder, x) for x in test_image_names]
test_label_paths = [os.path.join(yolo_test_label_folder, x.split('.')[0] + ".txt") for x in test_image_names]


for i_aeeg in range(len(test_image_paths)):
    image_name = test_image_names[i_aeeg]
    print("==== {} ====".format(image_name))

    image_path = test_image_paths[i_aeeg]
    label_path = test_label_paths[i_aeeg]

    # load image
    aeeg_image = Image.open(image_path)

    # load
    aeeg_label = load_coco_annotations_as_coords(file_path=label_path, image_h=1920, image_w=1920)

    # Draw ground-truth
    draw = ImageDraw.Draw(aeeg_image)
    fnt = ImageFont.truetype("arial.ttf", 15)
    for label in aeeg_label:
        class_id = label['class_id']
        coords = (label['x1'], label['y1'], label['x2'], label['y2'])
        if class_id == 0:
            draw.rectangle(coords, outline='red', width=3)
        elif class_id == 1:
            draw.rectangle(coords, outline='blue', width=3)

    # get prediction of this image
    prediction_pkl_file_name = image_name.split('.')[0] + '.pkl'
    prediction_pkl_file_path = os.path.join(prediction_pkl_folder, prediction_pkl_file_name)

    with open(prediction_pkl_file_path, 'rb') as f:
        predictions = pickle.load(f)

    for prediction in predictions:
        class_id = prediction[0]
        coords = (
            prediction[1],
            prediction[2],
            prediction[3],
            prediction[4]
        )
        conf = prediction[5]

        if class_id == 0:
            draw.rectangle(coords, outline='purple', width=2)
        elif class_id == 1:
            draw.rectangle(coords, outline='green', width=2)
        draw.text((coords[0], coords[1]), "{:.2f}".format(conf), font=fnt, fill='black')

    # Save the drawn image
    aeeg_image.save(
        os.path.join(
            vis_save_folder,
            image_name
        )
    )







