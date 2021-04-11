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

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
import os, json, cv2, random
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import output_to_kitti
import numpy as np
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
from detectron2.utils.logger import setup_logger
setup_logger()

import pycocotools
import coco_io
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader

from LossEvalHook import LossEvalHook
from detectron2.data import DatasetMapper

import pickle
import os

import pycocotools.mask as rletools

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode


########### CUSTOM data reader
def get_kitti_dataset_train(path_list):
    MAX_ITER = 10
    '''
    path_list = "["/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/0000.txt", 
                  "/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/0001.txt",
                  "/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/0002.txt",
                  ...
                  ]"
    '''
    
    img_dir = "/home/group00/mcv/datasets/KITTI-MOTS/training/image_02"
    
    labels_dir = path_list[0][:-8] # same for all: /home/group00/mcv/datasets/KITTI-MOTS/instances_txt/
    folders_list = [] # [0000, 0001, 0002, ...]
    all_files_list_dicts = []
    for path in path_list[:-4]:
        folder=path[-8:-4]+"/"
        folders_list.append(folder)
        dict_files_in_path = coco_io.load_txt(path) # returns a dictionary with 0,1,2,3... as keys
        all_files_list_dicts.append(dict_files_in_path)


    dataset_dicts = []
    for idx,all_files in tqdm(enumerate(all_files_list_dicts)):
        for key, value in all_files.items():
            record={}
            filename = str(key).zfill(6)+".png"
            #print(filename)
            img_filename = os.path.join(img_dir,folders_list[idx], filename) # TODO: get propper img_dir and folder
            height, width = cv2.imread(img_filename).shape[:2]

            record["file_name"] = img_filename
            record["image_id"] = str(key).zfill(4)
            record["height"] = height
            record["width"] = width

            classes = ['Pedestrian','Car']

            objs=[]

            for objects in value:
                class_id = objects.class_id
                instance_id = objects.track_id
                bbox = pycocotools.mask.toBbox(objects.mask)
                mask = objects.mask
                
                if class_id == 1 or class_id == 2:
                    obj = {
                        "type": 'Car' if class_id==1 else 'Pedestrian',
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 1 if class_id==1 else 0,
                        "segmentation": mask
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
    
    ############################ HERE WE DO MOTSChallenge #############################

    img_dir = "/home/group00/mcv/datasets/MOTSChallenge/train/images"
    
    labels_dir = path_list[0][:-8] # same for all: /home/group00/mcv/datasets/KITTI-MOTS/instances_txt/
    folders_list = [] # [0000, 0001, 0002, ...]
    all_files_list_dicts = []
    for path in path_list[-4:]:
        folder=path[-8:-4]+"/"
        folders_list.append(folder)
        dict_files_in_path = coco_io.load_txt(path) # returns a dictionary with 0,1,2,3... as keys
        all_files_list_dicts.append(dict_files_in_path)


    # dataset_dicts = []
    for idx,all_files in tqdm(enumerate(all_files_list_dicts)):
        for key, value in all_files.items():
            record={}
            filename = str(key).zfill(6)+".jpg"
            #print(filename)
            img_filename = os.path.join(img_dir,folders_list[idx], filename) # TODO: get propper img_dir and folder
            height, width = cv2.imread(img_filename).shape[:2]

            record["file_name"] = img_filename
            record["image_id"] = str(key).zfill(4)
            record["height"] = height
            record["width"] = width

            classes = ['Pedestrian','Car']

            objs=[]

            for objects in value:
                class_id = objects.class_id
                instance_id = objects.track_id
                bbox = pycocotools.mask.toBbox(objects.mask)
                mask = objects.mask
                
                if class_id == 1 or class_id == 2:
                    obj = {
                        "type": 'Car' if class_id==1 else 'Pedestrian',
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 1 if class_id==1 else 0,
                        "segmentation": mask
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

########### CUSTOM data reader val
def get_kitti_dataset_val(path_list):
    MAX_ITER = 10
    '''
    path_list = "["/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/0000.txt", 
                  "/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/0001.txt",
                  "/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/0002.txt",
                  ...
                  ]"
    '''
    
    img_dir = "/home/group00/mcv/datasets/KITTI-MOTS/training/image_02"
    
    labels_dir = path_list[0][:-8] # same for all: /home/group00/mcv/datasets/KITTI-MOTS/instances_txt/
    folders_list = [] # [0000, 0001, 0002, ...]
    all_files_list_dicts = []
    for path in path_list:
        folder=path[-8:-4]+"/"
        folders_list.append(folder)
        dict_files_in_path = coco_io.load_txt(path) # returns a dictionary with 0,1,2,3... as keys
        all_files_list_dicts.append(dict_files_in_path)


    dataset_dicts = []
    for idx,all_files in tqdm(enumerate(all_files_list_dicts)):
        for key, value in all_files.items():
            record={}
            filename = str(key).zfill(6)+".png"
            #print(filename)
            img_filename = os.path.join(img_dir,folders_list[idx], filename) # TODO: get propper img_dir and folder
            height, width = cv2.imread(img_filename).shape[:2]

            record["file_name"] = img_filename
            record["image_id"] = str(key).zfill(4)
            record["height"] = height
            record["width"] = width

            classes = ['Pedestrian','Car']

            objs=[]

            for objects in value:
                class_id = objects.class_id
                instance_id = objects.track_id
                bbox = pycocotools.mask.toBbox(objects.mask)
                mask = objects.mask
                
                if class_id == 1 or class_id == 2:
                    obj = {
                        "type": 'Car' if class_id==1 else 'Pedestrian',
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 1 if class_id==1 else 0,
                        "segmentation": mask
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts


# Dataset registration
base_path = "/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/"
test_paths = [base_path+str(i).zfill(4)+'.txt' for i in range(19,21)]
# TODO: define different folders for train and val (according to official splits)
# we are not testing so test doesn't matter for now
train_paths = [base_path+str(i).zfill(4)+'.txt' for i in [0,1,3,4,5,9,11,12,15,17,19,20]]
val_paths = [base_path+str(i).zfill(4)+'.txt' for i in [2,6,7,8,10,13,14,16,18]]

train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0002.txt")
train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0005.txt")
train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0009.txt")
train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0011.txt")

#train dataset
for d in [train_paths]:
    DatasetCatalog.register('train_kitti-mots', lambda d=d:get_kitti_dataset_train(d))
    MetadataCatalog.get('train_kitti-mots').set(thing_classes=['Pedestrian','Car'])

# #val dataset
# for d in [val_paths]:
#     DatasetCatalog.register('val_kitti-mots', lambda d=d:get_kitti_dataset_val(d))
#     MetadataCatalog.get('val_kitti-mots').set(thing_classes=['Pedestrian','Car'])

def visualization(configuration, image_paths, output_path):
    predictor = DefaultPredictor(configuration)
    if type(image_paths)==list:
        i = 0
        for image in image_paths:
            im = cv2.imread(image)
            output = predictor(im)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(configuration.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            v = v.draw_instance_predictions(output["instances"].to("cpu"))
            cv2.imwrite(f"{output_path}0{i}.png", v.get_image()[:, :, ::-1])
            i = i+1

def random_image_list(dataset_path):
    folders = os.listdir(dataset_path)
    paths = []
    for x in folders:
        paths.append(os.path.join(dataset_path, x))

    images_list = []
    for z in paths: 
        images_list.append(os.listdir(z))

    random_images = []
    for g in images_list:
        random_images.append(random.choice(g))

    random_paths = []
    for u in range (0,len(random_images)):
        random_paths.append(os.path.join(paths[u],random_images[u]))

    return random_paths


models = ["mask_rcnn_R_50_FPN_3x.yaml","mask_rcnn_R_50_C4_3x.yaml","mask_rcnn_R_50_DC5_3x.yaml","mask_rcnn_R_101_FPN_3x.yaml",  "mask_rcnn_R_101_C4_3x.yaml","mask_rcnn_R_101_DC5_3x.yaml"]
path = '/home/group00/working/week4/model_evaluation'
all_folders_paths = os.listdir(path)

models_to_evaluate = []

for folder in all_folders_paths:
    if folder !='Cityscapes':
        files = os.listdir(os.path.join(path,folder))
        for item in files:
            if item == 'model_final.pth':
                path_to_model = os.path.join(path,folder,item)
                output  = {'path':path_to_model, 'model_name':folder}
                models_to_evaluate.append(output)
    if folder== 'Cityscapes':
        sub_folders = os.listdir(os.path.join(path,folder))
        for subfolder in sub_folders:
            files = os.listdir(os.path.join(path,folder,subfolder))
            for item in files:
                if item == 'model_final.pth':
                    path_to_model = os.path.join(path,folder,subfolder,item)
                    output  = {'path':path_to_model, 'model_name':folder+'_'+subfolder}
                    models_to_evaluate.append(output)

print(models_to_evaluate)
# model_folder="COCO-InstanceSegmentation"
images_path = '/home/group00/mcv/datasets/KITTI-MOTS/training/image_02/'
paths_to_images = random_image_list(images_path)

for m in models_to_evaluate:
    model_path = m['path']
    model_name = m['model_name']
    output_path = f"/home/group00/working/week4/model_evaluation_images/{model_name}/"
    os.makedirs(output_path, exist_ok=True)
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("train_kitti-mots",)
    cfg.OUTPUT_DIR = output_path
    cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    predictor = DefaultPredictor(cfg)

    # paths= "/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0000/000005.png"
    visualization(configuration=cfg, image_paths=paths_to_images, output_path=output_path)
    

###############################################################################

# # Inference should use the config with parameters that are used in training
# # cfg now already contains everything we've set previously. We changed it a little bit for inference:
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)

# for d in dataset_dicts:    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
    
#     #x = outputs['instances'].get_fields()
    
#     #visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_train_metadata, scale=0.5)
#     # out = visualizer.draw_dataset_dict(out)
#     #out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
#     name = d["file_name"][-10:-4]
#     output_to_kitti(outputs, '../week2/my_experiment/data/'+name+'.txt')
#     #cv2.imwrite(f"{name}_globo.png",out.get_image()[:, :, ::-1])

