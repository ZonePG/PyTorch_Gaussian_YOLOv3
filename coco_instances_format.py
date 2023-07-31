category_name2id = {
    "Car": 1,
    "Cyclist": 2,
    "Pedestrian": 3,
}

dataset_format = {
    "info": None,
    "license": None,
    "images": [],
    "annotations": [],
    "categories": [
        {
            "supercategory": "Car",
            "id": 1,
            "name": "Car",
        },
        {
            "supercategory": "Cyclist",
            "id": 2,
            "name": "Cyclist",
        },
        {
            "supercategory": "Pedestrian",
            "id": 3,
            "name": "Pedestrian",
        }
    ],
}

image_format = {
    "license": None,
    "file_name": "",  # need to be filled
    "coco_url": None,
    "height": 0,      # need to be filled
    "width": 0,       # need to be filled
    "date_captured": None,
    "flickr_url": None,
    "id": 0,          # need to be filled
}

annotation_format = {
    "segmentation": None,
    "area": None,     # need to be filled
    "iscrowd": 0,
    "image_id": 0,    # need to be filled
    "bbox": [0, 0, 0, 0], # need to be filled, [upper_x, left_y, width, height]
    "category_id": 0, # need to be filled
    "id": 0,          # need to be filled, image_id << 2
}