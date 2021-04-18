# import tensorflow
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

from matplotlib.image import imread
import scipy.misc
from PIL import Image 
import pickle as pkl 

models = ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml","COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"]
models = ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]

path_to_data = '/home/group00/working/week5/img_task_D/output_data'
# model_folder="COCO-InstanceSegmentation"
# model_folder="COCO-Detection"

dataset_path = '/home/group00/working/week5/DATASETS/test2017/'
path_to_data=dataset_path

# paths = random_image_list(path)
paths = [os.path.join(path_to_data,x) for x in os.listdir(path_to_data)]
# paths = [os.path.join(path,'cat_base.png')]
# paths = [os.path.join(dataset_path,'000000330209.jpg')]


# paths = ['/home/group00/working/week5/DATASETS/test2017/000000330209.jpg']
# for i in range(0,10):
#     image_paths.append(os.path.join(dataset_path,random.choice(os.listdir(dataset_path))))

for m in models:
    # model_path = os.path.join(model_folder,m)
    model_path = m
    output_path = f"/home/group00/working/week5/test_outs_task_D/{m[:-5]}/laptop/"
    os.makedirs(output_path, exist_ok=True)
    label_index = 63

    # cfg = pretrained_config(model_url="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg = pretrained_config(model_url=model_path)

    #visualization
    configuration = cfg
    predictor = DefaultPredictor(configuration)
    image_paths = paths

    if type(image_paths)==list:
        i = 0
        classes_no_bkg_list = []
        for image in image_paths:
            image_name = image.split('/')[-1]

            im = cv2.imread(image)
            output = predictor(im)
            classes = output['instances'].pred_classes.cpu().numpy().tolist()

            if label_index in classes:
                with open(f'{output_path}/{image_name[:-3]}txt','w') as f:
                    f.write(str(output))
                
                idx_bird=classes.index(label_index)
                mask_array = output['instances'].pred_masks[idx_bird].cpu().numpy()
                classes = output['instances'].pred_classes.cpu().numpy()
                print(image,"classes=",classes)
                # cv2.imwrite('file.png',mask_array)
                data = Image.fromarray(mask_array)
                data.save(f'{output_path}/{image_name[:-4]}_mask.png')
                mask = cv2.imread(f'{output_path}/{image_name[:-4]}_mask.png',0)
                # inference without background
                no_bkg = cv2.bitwise_and(im, im, mask=mask)
                output_no_bkg = predictor(no_bkg)
                classes_no_bkg = output_no_bkg['instances'].pred_classes.cpu().numpy().tolist()
                if len(classes_no_bkg)>0:
                    classes_no_bkg_list.append((image_name,classes[idx_bird],classes_no_bkg[0]))
                else:
                    classes_no_bkg_list.append((image_name,classes[idx_bird], -1))

                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(configuration.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
                v = v.draw_instance_predictions(output["instances"].to("cpu"))
                cv2.imwrite(f"{output_path}{image_name[:-4]}_segm.png", v.get_image()[:, :, ::-1])
                cv2.imwrite(f"{output_path}{image_name}",im)
                cv2.imwrite(f"{output_path}{image_name[:-4]}_no_bkg.png",no_bkg)
                i = i+1

    with open('/home/group00/working/week5/test_outs_task_D/classes_bird_no_bkg.pkl','wb') as f:
        pkl.dump(classes_no_bkg_list,f)
# quit()
#     # paths= "/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0000/000005.png"
# visualization(configuration=cfg, image_paths=paths, output_path=output_path)
    