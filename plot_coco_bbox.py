import os
from pycocotools.coco import COCO
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
 
json_path = "/home/zonepg/datasets/kitti/annotations/instances_train2017.json"
img_path = "/home/zonepg/datasets/kitti/train2017"
 
# load coco data
coco = COCO(annotation_file=json_path)
 
# get all image index info
ids = list(sorted(coco.imgs.keys()))
print(f'number of images {len(ids)}')
 
# get all coco class labels
coco_classes = dict([(v["id"],v["name"]) for k,v in coco.cats.items()])
 
for img_id in ids[:10]:
    # 获取对应图像id的所有annotations idx信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
 
    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)
 
    # get image file name
    path = coco.loadImgs(img_id)[0]['file_name']
 
 
    # read image
    img = Image.open(os.path.join(img_path,path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    # draw box to image
    for target in targets:
        x,y,w,h = target['bbox']
        x1,y1,x2,y2 = x,y,int(x+w),int(y+h)
        draw.rectangle((x1,y1,x2,y2))
        draw.text((x1,y1), coco_classes[target['category_id']])
 
    # show image
    plt.imshow(img)
    plt.show()