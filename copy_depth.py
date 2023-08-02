import os
from glob import glob
import cv2

def fill_idx(idx, len=6, ext=".jpg"):
    return str(idx).zfill(len) + ext

src_path = "/home/zonepg/datasets/kitti/JPEGImages_dense"
dst_path = "/home/zonepg/datasets/kitti/JPEGImages_depth"

os.makedirs(dst_path, exist_ok=True)

img_list = sorted(glob(os.path.join(src_path, '*.png'), recursive=True))

for img in img_list:
    img_name = img.split(os.sep)[-1]
    img_id = img_name.split('.')[0]
    new_img_name = fill_idx(img_id)
    img = cv2.imread(img)
    cv2.imwrite(os.path.join(dst_path, new_img_name), img)

