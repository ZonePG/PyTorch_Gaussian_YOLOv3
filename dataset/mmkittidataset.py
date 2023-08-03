import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from utils.utils import *

import cv2


class MMKITTIDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir0='COCO', data_dir1='COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.data_dir0 = data_dir0
        self.data_dir1 = data_dir1
        self.json_file = json_file
        self.model_type = model_type
        self.coco0 = COCO(self.data_dir0+'annotations/'+self.json_file)
        self.coco1 = COCO(self.data_dir1+'annotations/'+self.json_file)
        self.ids = self.coco0.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco0.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        anno_ids = self.coco0.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco0.loadAnns(anno_ids)
        annotations_depth = self.coco1.loadAnns(anno_ids)

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True

        # load image and preprocess
        img_file0 = os.path.join(self.data_dir0, self.name,
                                '{:06}'.format(id_) + '.jpg')
        img0 = cv2.imread(img_file0)
        img_file1 = os.path.join(self.data_dir1, self.name,
                                '{:06}'.format(id_) + '.jpg')
        img1 = cv2.imread(img_file1)

        # if self.json_file == 'instances_val5k.json' and img is None:
        #     img_file = os.path.join(self.data_dir, 'train2017',
        #                             '{:06}'.format(id_) + '.jpg')
        #     img = cv2.imread(img_file)
        assert img0 is not None and img1 is not None

        img0, info_img = preprocess(img0, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)
        h, w, nh, nw, dx, dy = info_img
        img1 = img1[:, :, ::-1]
        img1 = cv2.resize(img1, (nw, nh))
        sized = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 127
        sized[dy:dy+nh, dx:dx+nw, :] = img1
        img1 = sized

        cv2.imwrite('img0.jpg', img0)
        cv2.imwrite('img1.jpg', img1)

        if self.random_distort:
            img0 = random_distort(img0, self.hue, self.saturation, self.exposure)
            img1 = random_distort(img1, self.hue, self.saturation, self.exposure)

        img0 = np.transpose(img0 / 255., (2, 0, 1))
        img1 = np.transpose(img1 / 255., (2, 0, 1))

        if lrflip:
            img0 = np.flip(img0, axis=2).copy()
            img1 = np.flip(img1, axis=2).copy()

        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img0, img1, padded_labels, info_img, id_
