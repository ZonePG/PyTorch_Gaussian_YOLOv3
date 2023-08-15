import sys
import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox

import matplotlib.pyplot as plt

# Choose config file for this demo
cfg_path = './config/gaussian_yolov3_kitti_eval.cfg'

# Specify checkpoint file which contains the weight of the model you want to use
ckpt_path = './checkpoints-kitti_final-batch8-gaussian/snapshot25000.ckpt'

# Path to the image file fo the demo
# image_path = './data/gaussian_yolov3/traffic_1.jpg'
image_path = '/home/zonepg/datasets/kitti_final/test2017/000000.jpg'

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
model = YOLOv3(model_config)

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


# Load image
img = cv2.imread(image_path)

# Preprocess image
img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
img = np.transpose(img / 255., (2, 0, 1))
img = torch.from_numpy(img).float().unsqueeze(0)

if gpu >= 0:
    # Send model to GPU
    img = Variable(img.type(torch.cuda.FloatTensor))
else:
    img = Variable(img.type(torch.FloatTensor))


# Inference
with torch.no_grad():
    outputs = model(img)
    outputs = postprocess(outputs, 3, confthre, nmsthre)

if outputs[0] is None:
    print("No Objects Deteted!!")
    sys.exit(0)

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
    img_raw, bboxes, label=classes, score=scores, label_names=kitti_class_names, sigma=sigmas, 
    sigma_scale_img=sigma_scale_img,
    sigma_scale_xy=2., sigma_scale_wh=2.,  # 2-sigma
    show_inner_bound=False,  # do not show inner rectangle for simplicity
    instance_colors=colors, linewidth=3)

fig.savefig('demo.png')
