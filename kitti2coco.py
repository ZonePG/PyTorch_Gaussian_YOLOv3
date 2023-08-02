import os
import argparse
from glob import glob
import cv2
from coco_instances_format import dataset_format, image_format, annotation_format, category_name2id
import json
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=bool, default=False)
    return parser.parse_args()


def label2cocojson(stage_file_list, stage, anno_path, label_path, src_images_path):
    dataset = copy.deepcopy(dataset_format)
    json_name = os.path.join(anno_path, "instances_" + stage + "2017.json")
    print(json_name)

    for file in stage_file_list:
        image_json = copy.deepcopy(image_format)
        image_json["file_name"] = file + ".jpg"
        image = cv2.imread(os.path.join(src_images_path, file + ".jpg"))
        image_json["height"] = image.shape[0]
        image_json["width"] = image.shape[1]
        image_json["id"] = int(file)
        dataset['images'].append(image_json)

        label_file = os.path.join(label_path, file + ".txt")
        label_list = open(label_file, "r").readlines()
        label_list = [label.strip().split(" ") for label in label_list]
        for i, obj in enumerate(label_list):
            xmin = float(obj[4])
            ymin = float(obj[5])
            xmax = float(obj[6])
            ymax = float(obj[7])
            width = xmax - xmin
            height = ymax - ymin

            annotation_json = copy.deepcopy(annotation_format)
            annotation_json["area"] = width * height
            annotation_json["image_id"] = int(file)
            annotation_json["bbox"] = [xmin, ymin, width, height]
            annotation_json["category_id"] = category_name2id[obj[0]]
            annotation_json["id"] = int(file) * 100 + i
            dataset["annotations"].append(annotation_json)

    with open(json_name, "w") as f:
        f.write(json.dumps(dataset))
    

def main():
    args = parse_args()

    if args.depth:
        src_path = "/home/zonepg/datasets/kitti_depth"
        src_images_path = os.path.join(src_path, "JPEGImages_depth")
    else:
        src_path = "/home/zonepg/datasets/kitti"
        src_images_path = os.path.join(src_path, "JPEGImages")

    train_dst_path = os.path.join(src_path, "train2017")
    val_dst_path = os.path.join(src_path, "val2017")
    test_dst_path = os.path.join(src_path, "test2017")
    os.makedirs(train_dst_path, exist_ok=True)
    os.makedirs(val_dst_path, exist_ok=True)
    os.makedirs(test_dst_path, exist_ok=True)

    anno_path = os.path.join(src_path, "annotations")
    os.makedirs(anno_path, exist_ok=True)

    label_path = os.path.join(src_path, "Labels")

    image_sets_path = os.path.join(src_path, "ImageSets", "Main")
    for stage in ["train", "val", "test"]:
        stage_path = os.path.join(image_sets_path, stage + ".txt")
        stage_file_list = open(stage_path, "r").readlines()
        stage_file_list = [file.strip() for file in stage_file_list]
        # copy image
        # for file in stage_file_list:
        #     src_file = os.path.join(src_images_path, file + ".jpg")
        #     image = cv2.imread(src_file)
        #     dst_file = os.path.join(src_path, stage + "2017", file + ".jpg")
        #     cv2.imwrite(dst_file, image)

        # make annotation
        label2cocojson(stage_file_list, stage, anno_path, label_path, src_images_path)


if __name__ == "__main__":
    main()
