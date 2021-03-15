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

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader




########### CUSTOM data reader
def get_kitti_dataset(all_files):
    MAX_ITER = 10
    '''
    img_dir: "/home/group00/mcv/datasets/KITTI/training/ - ideally
    '''
    img_dir = "/home/group00/mcv/datasets/KITTI/data_object_image_2/training/image_2/"
    labels_dir = "/home/group00/mcv/datasets/KITTI/training/label_2/"
    filenames = os.listdir(labels_dir)
    
    # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01

    dataset_dicts = []
    current_iter = 0
    for idx, item in tqdm(enumerate(all_files)):
        current_iter += 1
        #if current_iter > MAX_ITER:
        #   break
        # print("file=",item)
        if len(item)==0:
            break

        # print("item=",item)
        path_to_annotation = os.path.join(labels_dir, item)
        # print("path to ann=", path_to_annotation)
        with open(path_to_annotation, "r") as f:
            lines = f.read().split('\n')

        record = {}

        
        img_filename = os.path.join(img_dir, item[:-3]+"png")
        # print("filename=",img_filename)
        height, width = cv2.imread(img_filename).shape[:2]
        
        record["file_name"] = img_filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
              
        classes = ['Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc', 'DontCare']      
        objs = []
        for line in (lines):
            if len(line)!=0:
                # print("line=",line)
                annotations_list = line.split()
                obj = {
                    "type": annotations_list[0],
                    "truncated": float(annotations_list[1]),
                    "occluded": int(annotations_list[2]),
                    "alpha": float(annotations_list[3]),
                    "bbox": [float(annotations_list[4]), float(annotations_list[5]), float(annotations_list[6]), float(annotations_list[7])],
                    "dimensions" : [float(annotations_list[8]), float(annotations_list[9]), float(annotations_list[10])],
                    "location": [float(annotations_list[11]), float(annotations_list[12]), float(annotations_list[13])],
                    "rotation_y": float(annotations_list[14]),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": classes.index(annotations_list[0])
                    # "score":
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

# Dataset registration
for d in ["train_kitti","val_kitti"]:
    print("d=",d)
    # /home/group00/mcv/datasets/KITTI/training/label_2
    path = os.path.join("/home/group00/mcv/datasets/KITTI",d+".txt")
    with open(path, "r") as f:
        lines = f.read().split('\n')

    #DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    DatasetCatalog.register(d, lambda d=d:get_kitti_dataset(lines))
    MetadataCatalog.get(d).set(thing_classes=['Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc', 'DontCare'])

kitti_train_metadata = MetadataCatalog.get("train_kitti")

path = os.path.join("/home/group00/mcv/datasets/KITTI","train_kitti.txt")
with open(path, "r") as f:
    lines = f.read().split('\n')

dataset_dicts = get_kitti_dataset(lines)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite("file_globo.png",out.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.OUTPUT_DIR = "/home/group00/working/week2/models"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_kitti",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.DATASETS.TEST = ("val_kitti",) 
cfg.TEST.EVAL_PERIOD = 100
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = MyTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()

# #EVALUATION
evaluator = COCOEvaluator("val_kitti", cfg, False, output_dir="../week2/my_ex/")
val_loader = build_detection_test_loader(cfg, "val_kitti")
inference_on_dataset(trainer.model, val_loader, evaluator)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join("/home/group00/working/week2/", "model_good.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

path = os.path.join("/home/group00/mcv/datasets/KITTI","val_kitti.txt")
with open(path, "r") as f:
    lines = f.read().split('\n')
dataset_dicts = get_kitti_dataset(lines)

MAX_COUNT = 5
count = 0
for d in dataset_dicts:    
    count+=1
    if count > MAX_COUNT:
        break
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    
    #x = outputs['instances'].get_fields()
    
    visualizer = Visualizer(im[:, :, ::-1], metadata=kitti_train_metadata, scale=0.5)
    #out = visualizer.draw_dataset_dict(outputs)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    name = d["file_name"][-10:-4]
    output_to_kitti(outputs, '../week2/my_experiment/data3/'+name+'.txt')
    cv2.imwrite(f"{name}_{str(count)}_globo.png",out.get_image()[:, :, ::-1])


