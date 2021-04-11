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
    
    # ############################ HERE WE DO MOTSChallenge #############################

    # img_dir = "/home/group00/mcv/datasets/MOTSChallenge/train/images"
    
    # labels_dir = path_list[0][:-8] # same for all: /home/group00/mcv/datasets/KITTI-MOTS/instances_txt/
    # folders_list = [] # [0000, 0001, 0002, ...]
    # all_files_list_dicts = []
    # for path in path_list[-4:]:
    #     folder=path[-8:-4]+"/"
    #     folders_list.append(folder)
    #     dict_files_in_path = coco_io.load_txt(path) # returns a dictionary with 0,1,2,3... as keys
    #     all_files_list_dicts.append(dict_files_in_path)


    # # dataset_dicts = []
    # for idx,all_files in tqdm(enumerate(all_files_list_dicts)):
    #     for key, value in all_files.items():
    #         record={}
    #         filename = str(key).zfill(6)+".jpg"
    #         #print(filename)
    #         img_filename = os.path.join(img_dir,folders_list[idx], filename) # TODO: get propper img_dir and folder
    #         height, width = cv2.imread(img_filename).shape[:2]

    #         record["file_name"] = img_filename
    #         record["image_id"] = str(key).zfill(4)
    #         record["height"] = height
    #         record["width"] = width

    #         classes = ['Pedestrian','Car']

    #         objs=[]

    #         for objects in value:
    #             class_id = objects.class_id
    #             instance_id = objects.track_id
    #             bbox = pycocotools.mask.toBbox(objects.mask)
    #             mask = objects.mask
                
    #             if class_id == 1 or class_id == 2:
    #                 obj = {
    #                     "type": 'Car' if class_id==1 else 'Pedestrian',
    #                     "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
    #                     "bbox_mode": BoxMode.XYWH_ABS,
    #                     "category_id": 1 if class_id==1 else 0,
    #                     "segmentation": mask
    #                 }
    #                 objs.append(obj)

    #         record["annotations"] = objs
    #         dataset_dicts.append(record)

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

# train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0002.txt")
# train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0005.txt")
# train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0009.txt")
# train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0011.txt")

#train dataset
for d in [train_paths]:
    DatasetCatalog.register('train_kitti-mots', lambda d=d:get_kitti_dataset_train(d))
    MetadataCatalog.get('train_kitti-mots').set(thing_classes=['Pedestrian','Car'])

#val dataset
for d in [val_paths]:
    DatasetCatalog.register('val_kitti-mots', lambda d=d:get_kitti_dataset_val(d))
    MetadataCatalog.get('val_kitti-mots').set(thing_classes=['Pedestrian','Car'])

# models = ["mask_rcnn_R_50_FPN_3x.yaml","mask_rcnn_R_50_C4_3x.yaml","mask_rcnn_R_50_DC5_3x.yaml","mask_rcnn_R_101_FPN_3x.yaml",  "mask_rcnn_R_101_C4_3x.yaml","mask_rcnn_R_101_DC5_3x.yaml"]
models = ["Cityscapes/mask_rcnn_R_50_FPN.yaml"]

for to_evaluate in models:
    print('* '*30, to_evaluate , '* '*30)

    # Train
    lr = 0.0025
    EXPERIMENT_NAME = f"{to_evaluate[:-5]}_trained_noMOTS"
    OUTPUT_DIR = f"/home/group00/working/week4/model_evaluation/{EXPERIMENT_NAME}"


    print('Loading pre-trained models...')
    cfg = get_cfg()

    #Select model
    # cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{to_evaluate}"))
    cfg.merge_from_file(model_zoo.get_config_file(f"{to_evaluate}"))
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{to_evaluate}")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{to_evaluate}")

    #configure parameters
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.DATASETS.TRAIN = ("train_kitti-mots",)
    cfg.DATASETS.TEST = ("val_kitti-mots",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon).
    cfg.TEST.EVAL_PERIOD = 100

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        
        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1,LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                )
            ))
            return hooks

    # Training


    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    last_results = trainer.train()

    with open(f'{OUTPUT_DIR}/model_trained_results_last.pkl', 'wb') as f:
        pickle.dump(last_results, file=f)

    # Evaluate
    print('Evaluating...')

    evaluator = COCOEvaluator("val_kitti-mots", ("segm","bbox",), False, output_dir=OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "val_kitti-mots")
    results= inference_on_dataset(trainer.model, val_loader, evaluator)

    with open(f'{OUTPUT_DIR}/model_trained_results.pkl', 'wb') as f:
        pickle.dump(results, file=f)



