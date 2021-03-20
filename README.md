# M5 Project: Object Detection and Segmentation
## Group 00

### Team members:
* _Adam Szummer_ - a.szummer@gmail.com - [VanillaTiger](https://github.com/VanillaTiger)
* _Sergi Garcia Sarroca_ - sergi.garciasa@e-campus.uab.cat - [SoftonCV](https://github.com/SoftonCV)
* _Antoni Rodriguez Villegas_ - rv.antoni@hotmail.com - [antoniRodriguez](https://github.com/antoniRodriguez)



# Week 2: Hands on Detectron -- FasterRCNN, Retina, FineTuning. 

#### Dataset
MIT_Split and KITI

![](week2/Images/Dataset_Sample.jpeg)

The principal task is to run Faster RCN, Retina on MIT data split and evaluate it qualitatively and furthermore fine-tune faster rcnn on KITTI dataset.

Inside `week2/` are the files with implementation of Faster RCNN, Retina, FineTuning script as well as supporting scripts implemented in pytorch

Tasks to be done: 

    - [x] Tutorial on detectron2. 

    - [x] Use object detection models in inference: Faster R-CNN. 
    
    - [x] Use object detection models in inference: RetinaNet
    
    - [x] Train Faster R-CNN on KITTI dataset. 
    
    - [x] Review state of the art object detection. 

- Slides for the project: [T00-Google Slides] https://docs.google.com/presentation/d/1b1SwpNROnXxFXRYwuKHxv2l9IO0JhPoF3N-lLz-iT7M/edit?usp=sharing

- Link to the Overleaf article (non-editable): https://www.overleaf.com/read/dscbpxyptkgk

(a) Apply pre-trained Mask-RCNN models to KITTI-MOTS validation set. Trying different configurations:
  - [x] Number of layers
  - [x] Backbone configuration
  - [x] Use of Feature Pyramid Network
  - [x] Use of training data (COCO vs COCO+Cityscapes)
