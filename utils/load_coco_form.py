def coco_to_x1y1x2y2(coco_bbox_dictionary: dict, image_h: int, image_w: int) -> tuple:
    """

    Parameters
    ----------
    coco_bbox_dictionary: i.e {'class_id': 0.0, 'x_center': 0.11354166666666667, 'y_center': 0.028385416666666666, 'width': 0.005208333333333333, 'height': 0.0359375}
    image_h: 1920
    image_w: 1920

    Returns
    -------

    """
    x_center = coco_bbox_dictionary["x_center"]
    y_center = coco_bbox_dictionary["y_center"]
    width = coco_bbox_dictionary["width"]
    height = coco_bbox_dictionary["height"]

    x1 = (x_center - (width * 0.5)) * image_w
    x2 = (x_center + (width * 0.5)) * image_w
    y1 = (y_center - (height * 0.5)) * image_h
    y2 = (y_center + (height * 0.5)) * image_h

    return int(x1), int(y1), int(x2), int(y2)

def load_coco_annotations_as_coords(file_path: str, image_h: int, image_w: int) -> list:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        annotations.append({
            'class_id': int(class_id),
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })

    coord_annotations = []
    for annotation in annotations:
        coord_annotation = coco_to_x1y1x2y2(coco_bbox_dictionary=annotation, image_h=image_h, image_w=image_w)
        coord_annotations.append(
            {
                'class_id': annotation['class_id'],
                'x1': coord_annotation[0],
                'y1': coord_annotation[1],
                'x2': coord_annotation[2],
                'y2': coord_annotation[3],
            }
        )

    return coord_annotations

