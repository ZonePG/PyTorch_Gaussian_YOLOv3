import sys
import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from models.mmyolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox

import matplotlib.pyplot as plt

from glob import glob
import os
from tqdm import tqdm

exp = {
    "kitti_rgb": {
        "ckpt_path": "checkpoints-kitti_rgb-batch8-gaussian/snapshot50000.ckpt",
        "image_folder": "/home/zonepg/datasets/kitti/test2017",
        "output_folder": "visualize/kitti_rgb",
    },
    "kitti_low_0.2_noise": {
        "ckpt_path": "checkpoints-kitti_low_0.2_noise-batch8-gaussian/snapshot50000.ckpt",
        "image_folder": "/home/zonepg/datasets/kitti_low_0.2_noise/test2017",
        # "image_folder": "/home/zonepg/datasets/kitti/test2017",
        "output_folder": "visualize/kitti_low_0.2_noise",
    },
    "kitti_low_0.2_noise-mm": {
        "ckpt_path": "checkpoints-kitti_low_0.2_noise-batch8-gaussian-mm/snapshot36000.ckpt",
        "image_folder": "/home/zonepg/datasets/kitti_low_0.2_noise/test2017",
        "output_folder": "visualize/kitti_low_0.2_noise-mm",
    },
    "kitti_final": {
        "ckpt_path": "checkpoints-kitti_final-batch8-gaussian/snapshot50000.ckpt",
        "image_folder": "/home/zonepg/datasets/kitti_final/test2017",
        "output_folder": "visualize/kitti_final",
    },
    "kitti_final-mm": {
        "ckpt_path": "checkpoints-kitti_final-batch8-gaussian-mm/snapshot35000.ckpt",
        "image_folder": "/home/zonepg/datasets/kitti_final/test2017",
        "output_folder": "visualize/kitti_final-mm",
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='config/gaussian_yolov3_kitti_eval.cfg')
    parser.add_argument('--exp_name', type=str, default='kitti_rgb')
    parser.add_argument('--depth_image_folder', type=str, default='/home/zonepg/datasets/kitti_depth/test2017')
    parser.add_argument('--multimodal', action='store_true', default=False, help='train with rgb and depth image')
    return parser.parse_args()


def main():
    args = parse_args()
    # Choose config file for this demo
    cfg_path = args.cfg_path

    # Specify checkpoint file which contains the weight of the model you want to use
    ckpt_path = exp[args.exp_name]["ckpt_path"]

    # Path to the image file fo the demo
    # image_path = './data/gaussian_yolov3/traffic_1.jpg'
    image_folder = exp[args.exp_name]["image_folder"]
    depth_image_folder = args.depth_image_folder
    output_folder = exp[args.exp_name]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    # Detection threshold
    detect_thresh = 0.3

    # Use CPU if gpu < 0 else use GPU
    gpu = -1

    # Load configratio parameters for this demo
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_config = cfg['MODEL']
    imgsize = cfg['TEST']['IMGSIZE']
    confthre = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']
    gaussian = cfg['MODEL']['GAUSSIAN']

    # if detect_thresh is not specified, the parameter defined in config file is used
    if detect_thresh:
        confthre = detect_thresh


    # Load model
    if not args.multimodal:
        model = YOLOv3(cfg['MODEL'])
    else:
        model = MMYOLOv3(cfg['MODEL'])

    # Load weight from the checkpoint
    print("loading checkpoint %s" % (ckpt_path))
    state = torch.load(ckpt_path)

    if 'model_state_dict' in state.keys():
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    model.eval()

    if gpu >= 0:
        # Send model to GPU
        model.cuda()


    image_list = sorted(glob(os.path.join(image_folder, '*.jpg'), recursive=True))
    depth_image_list = sorted(glob(os.path.join(depth_image_folder, '*.jpg'), recursive=True))
    for (image_path, depth_image_path) in zip(image_list, depth_image_list):
        print(image_path)
        # Load image
        img = cv2.imread(image_path)
        depth_img = cv2.imread(depth_image_path)

        # Preprocess image
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        depth_img_raw = depth_img.copy()[:, :, ::-1].transpose((2, 0, 1))
        depth_img, info_img = preprocess(depth_img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        depth_img = np.transpose(depth_img / 255., (2, 0, 1))
        depth_img = torch.from_numpy(depth_img).float().unsqueeze(0)

        if gpu >= 0:
            # Send model to GPU
            img = Variable(img.type(torch.cuda.FloatTensor))
            depth_img = Variable(depth_img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))
            depth_img = Variable(depth_img.type(torch.FloatTensor))


        # Inference
        with torch.no_grad():
            if not args.multimodal:
                outputs = model(img)
            else:
                outputs = model(img, depth_img)
            outputs = postprocess(outputs, 3, confthre, nmsthre)

        if outputs[0] is None:
            print("No Objects Deteted!!")
            # sys.exit(0)
            continue

        # Visualize detected bboxes
        kitti_class_names, kitti_class_ids, kitti_class_colors = get_kitti_label_names()

        bboxes = list()
        classes = list()
        scores = list()
        colors = list()
        sigmas = list()

        for output in outputs[0]:
            x1, y1, x2, y2, conf, cls_conf, cls_pred = output[:7]
            if gaussian:
                sigma_x, sigma_y, sigma_w, sigma_h = output[7:]
                sigmas.append([sigma_x, sigma_y, sigma_w, sigma_h])

            cls_id = kitti_class_ids[int(cls_pred)]
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)
            classes.append(cls_id)
            scores.append(cls_conf * conf)
            colors.append(kitti_class_colors[int(cls_pred)])
            
            # image size scale used for sigma visualization
            h, w, nh, nw, _, _ = info_img
            sigma_scale_img = (w / nw, h / nh)
            
        fig, ax = vis_bbox(
            img_raw, bboxes, label=classes, score=scores, label_names=kitti_class_names, sigma=[], 
            sigma_scale_img=sigma_scale_img,
            sigma_scale_xy=2., sigma_scale_wh=2.,  # 2-sigma
            show_inner_bound=False,  # do not show inner rectangle for simplicity
            instance_colors=colors, linewidth=3)

        fig.savefig(os.path.join(output_folder, image_path.split(os.sep)[-1]))
        plt.close()

if __name__ == "__main__":
    main()
