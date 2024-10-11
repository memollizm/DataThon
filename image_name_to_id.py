import json
from pycocotools.coco import COCO


annotation_file_path = 'annotations/train.json'

coco_ann = COCO(annotation_file=annotation_file_path)

imgfile2imgid = {coco_ann.imgs[i]['file_name']: i for i in coco_ann.imgs.keys()}


with open('train_id.json', 'w') as f:
   json.dump(imgfile2imgid, f)