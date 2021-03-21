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

            classes = ['Car', 'Pedestrian']

            objs=[]

            for objects in value:
                class_id = objects.class_id
                instance_id = objects.track_id
                bbox = pycocotools.mask.toBbox(objects.mask)
                
                if class_id == 1 or class_id == 2:
                    obj = {
                        "type": 'Car' if class_id==1 else 'Pedestrian',
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 0 if class_id==1 else 1
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

            classes = ['Car', 'Pedestrian']

            objs=[]

            for objects in value:
                class_id = objects.class_id
                instance_id = objects.track_id
                bbox = pycocotools.mask.toBbox(objects.mask)
                
                if class_id == 1 or class_id == 2:
                    obj = {
                        "type": 'Car' if class_id==1 else 'Person',
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 2 if class_id==1 else 0
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

            classes = ['Person','NA','Car']

            objs=[]

            for objects in value:
                class_id = objects.class_id
                instance_id = objects.track_id
                bbox = pycocotools.mask.toBbox(objects.mask)
                
                if class_id == 1 or class_id == 2:
                    obj = {
                        "type": 'Car' if class_id==1 else 'Person',
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 2 if class_id==1 else 0
                    }
                    
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts



# Dataset registration
base_path = "/home/group00/mcv/datasets/KITTI-MOTS/instances_txt/"
# train_paths = [base_path+str(i).zfill(4)+'.txt' for i in range(17)]
# val_paths = [base_path+str(i).zfill(4)+'.txt' for i in range(17,19)]
test_paths = [base_path+str(i).zfill(4)+'.txt' for i in range(19,21)]
# TODO: define different folders for train and val (according to official splits)
# we are not testing so test doesn't matter for now
train_paths = [base_path+str(i).zfill(4)+'.txt' for i in [0,1,3,4,5,9,11,12,15,17,19,20]]
val_paths = [base_path+str(i).zfill(4)+'.txt' for i in [2,6,7,8,10,13,14,16,18]]

train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0002.txt")
train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0005.txt")
train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0009.txt")
train_paths.append("/home/group00/mcv/datasets/MOTSChallenge/train/instances_txt/0011.txt")


# split_names = ['train_kitti-mots', 'val_kitti-mots']
# for d,split_name in zip([train_paths,val_paths,test_paths],split_names):
#     DatasetCatalog.register(split_name, lambda d=d:get_kitti_dataset(d))
#     MetadataCatalog.get(split_name).set(thing_classes=['Car',
#                      'Pedestrian'])


# kitti_train_metadata = MetadataCatalog.get("train_kitti-mots")


# dataset_dicts = get_kitti_dataset_train(train_paths)

# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_train_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imwrite("kitti-mots.png",out.get_image()[:, :, ::-1])

import os

OUTPUT_DIR = "/home/group00/working/week3/inference_test_antoni"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#train dataset
for d in [train_paths]:
    DatasetCatalog.register('train_kitti-mots', lambda d=d:get_kitti_dataset_train(d))
    MetadataCatalog.get('train_kitti-mots').set(thing_classes=['Person','NA','Car'])

#val dataset
for d in [val_paths]:
    DatasetCatalog.register('val_kitti-mots', lambda d=d:get_kitti_dataset_val(d))
    MetadataCatalog.get('val_kitti-mots').set(thing_classes=['Person','NA','Car'])

# Pre-trained
print('Loading pre-trained models...')
cfg = get_cfg()

#Select model

'''model_zoo_yml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_yml))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_yml)'''

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.9  # set threshold for this model

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")


predictor = DefaultPredictor(cfg)

# Evaluate
print('Evaluating...')
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("val_kitti-mots", ("bbox",), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "val_kitti-mots")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

'''cfg = get_cfg()

# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))


#cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))


#train dataset
for d in [train_paths]:
    DatasetCatalog.register('train_kitti-mots', lambda d=d:get_kitti_dataset_train(d))
    MetadataCatalog.get('train_kitti-mots').set(thing_classes=['Person','NA','Car'])

#val dataset
for d in [val_paths]:
    DatasetCatalog.register('val_kitti-mots', lambda d=d:get_kitti_dataset_val(d))
    MetadataCatalog.get('val_kitti-mots').set(thing_classes=['Person','NA','Car'])


MetadataCatalog.get("val_kitti-mots").set(thing_classes=['Person','NA','Car'])
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#Loading weights of pretrained model. 
cfg.DATASETS.TRAIN = ("train_kitti-mots",)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.DATASETS.TEST = ("val_kitti-mots",)
cfg.OUTPUT_DIR = OUTPUT_DIR
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.TEST.EVAL_PERIOD = 100
# class MyTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
#         return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.insert(-1,LossEvalHook(
#             cfg.TEST.EVAL_PERIOD,
#             self.model,
#             build_detection_test_loader(
#                 self.cfg,
#                 self.cfg.DATASETS.TEST[0],
#                 DatasetMapper(self.cfg,True)
#             )
#         ))
#         return hooks
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
print(trainer.test)
print('Something weird is happening down there')
# evaluator = COCOEvaluator(dataset + '_val', cfg, False, output_dir=join(dataset, model,'eval'))
# val_loader = build_detection_test_loader(cfg, dataset + '_val')
# inference_on_dataset(trainer.model, val_loader, evaluator)

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("val_kitti-mots", ("bbox",), False, output_dir="../week3/inference/")
val_loader = build_detection_test_loader(cfg, "val_kitti-mots")
inference_on_dataset(predictor.model, val_loader, evaluator)'''


# predictor = DefaultPredictor(cfg)

# predictions_path = "/home/group00/working/week3/inference/predictions.pkl"

# predictions = []
# for i, input_test in enumerate(get_kitti_dataset_val(val_paths)):
#     img_path = input_test['file_name']
#     img = cv2.imread(img_path)
#     prediction = predictor(img)
#     predictions.append(prediction)
# pickle.dump(predictions, open(predictions_path, "wb"))

# print('Predictions length ' + str(len(predictions)))
# print('Inputs length ' + str(len(get_kitti_dataset_val(val_paths))))

# # Evaluation
# print('Evaluating......')
# evaluator = COCOEvaluator("val_kitti-mots", cfg, False, output_dir="../week3/inference/")
# evaluator.reset()
# evaluator.process(get_kitti_dataset_val(val_paths), predictions)
# evaluator.evaluate()


# # #EVALUATION
# evaluator = COCOEvaluator("val_kitti-mots", cfg, False, output_dir="../week3/inference/")
# val_loader = build_detection_test_loader(cfg, "val_kitti-mots")
# inference_on_dataset(predictor.model, val_loader, evaluator)

# evaluator.reset()
# evaluator.process(kitti_test(), predictions)
# evaluator.evaluate()

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = MyTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()







# SHOULD BE OK WITH THE CODE ABOVE THIS PART
'''
#EVALUATION
evaluator = COCOEvaluator("val_kitti", cfg, False, output_dir="../week2/my_ex/")
val_loader = build_detection_test_loader(cfg, "val_kitti")
inference_on_dataset(trainer.model, val_loader, evaluator)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

path = os.path.join("/home/group00/mcv/datasets/KITTI","val_kitti.txt")
with open(path, "r") as f:
    lines = f.read().split('\n')
dataset_dicts = get_kitti_dataset(lines)

for d in dataset_dicts:    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    
    #x = outputs['instances'].get_fields()
    
    #visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_train_metadata, scale=0.5)
    # out = visualizer.draw_dataset_dict(out)
    #out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    name = d["file_name"][-10:-4]
    output_to_kitti(outputs, '../week2/my_experiment/data/'+name+'.txt')
    #cv2.imwrite(f"{name}_globo.png",out.get_image()[:, :, ::-1])


'''