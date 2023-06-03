import os
from PIL import Image, ImageDraw, ImageFont
import pickle
from utils.load_coco_form import load_coco_annotations_as_coords

# root info, never change.
yolo_dataset_root = "dataset_yolo_epi"
vis_save_root = "vis_dataset"

# makedir vis image saving folder
vis_save_folder = os.path.join(vis_save_root, "vis_" + yolo_dataset_root)
os.makedirs(vis_save_folder, exist_ok=True)
os.makedirs(os.path.join(vis_save_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(vis_save_folder, "val"), exist_ok=True)
os.makedirs(os.path.join(vis_save_folder, "test"), exist_ok=True)

# get input paths
yolo_train_image_folder = os.path.join(yolo_dataset_root, "images", "train")
yolo_train_label_folder = os.path.join(yolo_dataset_root, "labels", "train")
train_image_names = os.listdir(yolo_train_image_folder)
train_image_paths = [os.path.join(yolo_train_image_folder, x) for x in train_image_names]
train_label_paths = [os.path.join(yolo_train_label_folder, x[:-4] + ".txt") for x in train_image_names]

yolo_val_image_folder = os.path.join(yolo_dataset_root, "images", "val")
yolo_val_label_folder = os.path.join(yolo_dataset_root, "labels", "val")
val_image_names = os.listdir(yolo_val_image_folder)
val_image_paths = [os.path.join(yolo_val_image_folder, x) for x in val_image_names]
val_label_paths = [os.path.join(yolo_val_label_folder, x[:-4] + ".txt") for x in val_image_names]

yolo_test_image_folder = os.path.join(yolo_dataset_root, "images", "test")
yolo_test_label_folder = os.path.join(yolo_dataset_root, "labels", "test")
test_image_names = os.listdir(yolo_test_image_folder)
test_image_paths = [os.path.join(yolo_test_image_folder, x) for x in test_image_names]
test_label_paths = [os.path.join(yolo_test_label_folder, x[:-4] + ".txt") for x in test_image_names]


for i_aeeg in range(len(train_image_paths)):
    image_name = train_image_names[i_aeeg]
    print("==== {} ====".format(image_name))

    image_path = train_image_paths[i_aeeg]
    label_path = train_label_paths[i_aeeg]

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
            draw.rectangle(coords, outline='purple', width=3)

    # Save the drawn image
    aeeg_image.save(
        os.path.join(
            vis_save_folder,
            "train",
            image_name
        )
    )




for i_aeeg in range(len(val_image_paths)):
    image_name = val_image_names[i_aeeg]
    print("==== {} ====".format(image_name))

    image_path = val_image_paths[i_aeeg]
    label_path = val_label_paths[i_aeeg]

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
            draw.rectangle(coords, outline='purple', width=3)

    # Save the drawn image
    aeeg_image.save(
        os.path.join(
            vis_save_folder,
            "val",
            image_name
        )
    )





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
            draw.rectangle(coords, outline='purple', width=3)

    # Save the drawn image
    aeeg_image.save(
        os.path.join(
            vis_save_folder,
            "test",
            image_name
        )
    )




