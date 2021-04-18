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

models = ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"]
path =  "/home/group00/mcv/datasets/out_of_context"
# folder = "COCO-InstanceSegmentation"
images = os.listdir(path)
images_paths = []

for img in images: 
    image_path = os.path.join(path, img)
    images_paths.append(image_path)

output_dir = f"/home/group00/working/week5/"
for url in models:
    output_dir = f"/home/group00/working/week5/task_a_outs/{url[:-5]}/"
    os.makedirs(output_dir, exist_ok=True)
    cfg = pretrained_config(model_url=url)
    visualization(configuration=cfg, image_paths=images_paths, output_path=output_dir)

