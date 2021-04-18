import tensorflow
import detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt 
from pretrained_utils import *
import pickle

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

from matplotlib.image import imread
from tqdm import tqdm
import scipy.misc
from PIL import Image  

models = ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml","COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"]
path = '/home/group00/working/week5/img_task_C' # Change path in order to run task C or B
paths = [os.path.join(path,x) for x in os.listdir(path)]


evaluate_differences = False
save_pkl = False
if not evaluate_differences:
    for idx,model_path in enumerate(models):
        output_path = f"/home/group00/working/week5/test_outs_task_C/{model_path[:-5]}/" # Change path in order to run task C or B
        os.makedirs(output_path, exist_ok=True)
        
        cfg = pretrained_config(model_url=model_path)

        configuration = cfg
        predictor = DefaultPredictor(configuration)
        image_paths = paths

        classes = []
        images = []
        outputs = []
        if type(image_paths)==list:
            i = 0
            for image in tqdm(image_paths):
                im = cv2.imread(image)
                output = predictor(im)
                numpy_classes = output["instances"].to("cpu").pred_classes.numpy()

                outputs.append(output["instances"].to("cpu"))
                images.append(image)
                classes.append(numpy_classes)
                classes.append(numpy_classes)

                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(configuration.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
                v = v.draw_instance_predictions(output["instances"].to("cpu"))
                cv2.imwrite(output_path+image[-16:-4]+".png", v.get_image()[:, :, ::-1])
                i = i+1

            if save_pkl:
                dictionary = {}
                dictionary['classes'] = classes
                dictionary['instances'] = outputs
                dictionary['images'] = images

                f = open('week5/keyboard_task_C_outs'+str(idx)+'.pkl', 'wb') # Change path in order to run task C or B
                pickle.dump(dictionary, f)
                f.close()
else:
    f = open('/home/group00/working/week5/keyboard_task_b_outs.pkl', 'rb') # Change path in order to run task C or B
    keyboard_results_dictionary = pickle.load(f)
    f.close()

    f = open('/home/group00/working/week5/unmodified_task_b_outs.pkl', 'rb') # Change path in order to run task C or B
    original_results_dictionary = pickle.load(f)
    f.close()

    n_images = len(keyboard_results_dictionary['images'])
    dictionary_differences = {}
    dictionary_no_dining_table = {}
    for idx in tqdm(range(n_images)):
        original_n_detections = len(original_results_dictionary['classes'][idx])
        new_n_detections = len(keyboard_results_dictionary['classes'][idx])
        difference = new_n_detections - original_n_detections
        dictionary_differences[original_results_dictionary['images'][idx][-16:-4]] = difference

        dining_table_class = 60
        save_images = False
        if dining_table_class not in keyboard_results_dictionary['classes'][idx]:
            dictionary_no_dining_table[len(dictionary_no_dining_table)] = original_results_dictionary['images'][idx]
       
        if save_images:
            image_name = keyboard_results_dictionary['images'][idx][-16:-4]+'.png'
            maskRcnn = 'COCO-Detection/faster_rcnn_R_50_FPN_3x/'
            fasterRcnn = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/'
            keyboard = '/home/group00/working/week5/test_outs_task_B_keyboard_dining_table_/' # Change path in order to run task C or B
            unmodified = '/home/group00/working/week5/test_outs_task_B_unmodified_dining_table_/' # Change path in order to run task C or B
            keyboard_detection_maskRcnn = cv2.imread(keyboard+maskRcnn+image_name)
            unmodified_detection_maskRcnn = cv2.imread(unmodified+maskRcnn+image_name)
            keyboard_detection_fasterRcnn = cv2.imread(keyboard+fasterRcnn+image_name)
            unmodified_detection_fasterRcnn = cv2.imread(unmodified+fasterRcnn+image_name)

            output_folder = '/home/group00/working/week5/task_b_comparison_results/'
            cv2.imwrite(output_folder+image_name[:-4]+'_fasterRcnn_Keyboard.png', keyboard_detection_fasterRcnn)
            cv2.imwrite(output_folder+image_name[:-4]+'_fasterRcnn_unmodified.png', unmodified_detection_fasterRcnn)
            cv2.imwrite(output_folder+image_name[:-4]+'_maskRcnn_Keyboard.png', keyboard_detection_maskRcnn)
            cv2.imwrite(output_folder+image_name[:-4]+'_maskRcnn_unmodified.png', unmodified_detection_maskRcnn)


    f = open('week5/differences_n_detections.pkl', 'wb')
    pickle.dump(dictionary_differences, f)
    f.close()

    f = open('week5/no_dining_table.pkl', 'wb')
    pickle.dump(dictionary_no_dining_table, f)
    f.close()
        

quit()
    # paths= "/home/group00/mcv/datasets/KITTI-MOTS/testing/image_02/0000/000005.png"
visualization(configuration=cfg, image_paths=paths, output_path=output_path)
    
