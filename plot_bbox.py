import cv2
import numpy as np

image = cv2.imread("/home/zonepg/datasets/KITTI/JPEGImages/001785.jpg")

xmin = 712
ymin = 143
xmax = 810
ymax = 307
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
cv2.imwrite('img.jpg', image)
