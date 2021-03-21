import tensorflow
import detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()


#set up configuration
cfg = get_cfg()
#Adding the configuration to a desired model
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#Loading weights of pretrained model. 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

path = '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/'
path2 = '/home/group00/working/week3/retina_task1/'
folders = os.listdir(path)

paths = []
for x in folders:
    paths.append(os.path.join(path, x))

print('Len paths: ', len(paths))

images_list = []
for z in paths: 
    images_list.append(os.listdir(z))

print('Len images_list: ' , len(images_list))

random_images = []
for g in images_list:
    random_images.append(random.choice(g))

print(random_images)

random_paths = []
for u in range (0,len(random_images)):
    random_paths.append(os.path.join(paths[u],random_images[u]))

print('Random_paths: ', len(random_paths))

print(random_paths)

random_paths = ['/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0023/000329.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0022/000157.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0004/000156.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0018/000123.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0017/000200.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0001/000047.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0008/000027.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0019/000028.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0010/000073.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0025/000043.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0024/000085.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0007/000195.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0026/000044.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0012/000584.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0006/000079.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0014/000624.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0011/000511.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0009/000190.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0003/000149.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0013/000086.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0015/000060.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0021/000028.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0027/000067.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0002/000126.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0000/000456.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0020/000091.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0016/000446.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0005/000285.png', '/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0028/000048.png']

i = 0
for image in random_paths:
    im = cv2.imread(image)
    output = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imwrite(f"{path2}0{i}.png", v.get_image()[:, :, ::-1])
    i = i+1


