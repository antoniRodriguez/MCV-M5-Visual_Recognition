import tensorflow
import detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt 
from pretrained_utils import *

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()


models = ["mask_rcnn_R_50_FPN_3x.yaml","mask_rcnn_R_50_C4_3x.yaml","mask_rcnn_R_50_DC5_3x.yaml","mask_rcnn_R_101_FPN_3x.yaml",  "mask_rcnn_R_101_C4_3x.yaml","mask_rcnn_R_101_DC5_3x.yaml"]
path = '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/'
model_folder="COCO-InstanceSegmentation"

paths = random_image_list(path)

for m in models:
    model_path = os.path.join(model_folder,m)
    output_path = f"/home/group00/working/week4/test_outs/{m[:-5]}/"
    os.makedirs(output_path, exist_ok=True)

    # cfg = pretrained_config(model_url="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg = pretrained_config(model_url=model_path)

    # paths= "/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0000/000005.png"
    visualization(configuration=cfg, image_paths=paths, output_path=output_path)
    

