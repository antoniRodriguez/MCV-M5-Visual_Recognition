# import tensorflow
import detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt 

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
setup_logger()


def model_configuration(model_url, learning_rate, max_iter):

    model_a = os.path.join("COCO-InstanceSegmentation", model_url)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_a))
    cfg.DATASETS.TRAIN = ("kitti-mots",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = learning_rate 
    cfg.SOLVER.MAX_ITER = max_iter  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.INPUT.MASK_FORMAT = 'rle'
    # cfg.INPUT.MASK_FORMAT='bitmask'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print('Training...')
    trainer.train()

    # EVALUATION
    print('Evaluating...')
    evaluator = COCOEvaluator("kitti-mots", cfg, False, output_dir="./Search/")
    val_loader = build_detection_test_loader(cfg, "kitti-mots")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)

    return results

def pretrained_config(model_url, threshold=0.5):
    #set up configuration
    cfg = get_cfg()
    #Adding the configuration to a desired model
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold   # set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    #Loading weights of pretrained model. 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
    predictor = DefaultPredictor(cfg)
    return cfg

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
        
    elif type(image_paths)==str:
        pred= 'Predicted_image'
        im = cv2.imread(image_paths)
        output = predictor(im)
        print(type(output))
        print(output)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(configuration.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        cv2.imwrite(f"{output_path}0{pred}.png", v.get_image()[:, :, ::-1])

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


def evaluation(output_path, configuation, eval_dataset="val_kitti-mots", eval_concept=("segm",)):
    #EVALUATION
    evaluator = COCOEvaluator(eval_dataset, eval_concept, False, output_dir=output_path)
    val_loader = build_detection_test_loader(configuration, "val_kitti-mots")
    inference_on_dataset(predictor.model, val_loader, evaluator)


            