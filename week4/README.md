# M5 Project: Object Detection and Segmentation
## Group 00

### Team members:
* _Adam Szummer_ - a.szummer@gmail.com - [VanillaTiger](https://github.com/VanillaTiger)
* _Sergi Garcia Sarroca_ - sergi.garciasa@e-campus.uab.cat - [SoftonCV](https://github.com/SoftonCV)
* _Antoni Rodriguez Villegas_ - rv.antoni@hotmail.com - [antoniRodriguez](https://github.com/antoniRodriguez)

# Week 4: Hands on Detectron -- Object Detection and Instance Segmentation. 

Main tasks:

(a) Apply pre-trained Mask-RCNN models to KITTI-MOTS validation set. Trying different configurations:

• Get quantitative and qualitative results for both object detection and object segmentation

• Analyze the different configurations depending on:

  - [x] Number of layers
  - [x] Backbone configuration
  - [x] Use of Feature Pyramid Network
  - [x] Use of training data (COCO vs COCO+Cityscapes)

 (b) Train Mask-RCNN model on KITTI-MOTS and MOTSChallenge training sets and evaluate on KITTIMOTS validation set.
 
• Get quantitative and qualitative results for both object detection and object segmentation

• Compare REsults depending on the training data used:
 
  - [x] COCO (Task a)
  - [x] COCO + Cityscapes (Task a)
  - [x] COCO + KITTI-MOTS
  - [x] COCO + KITTI-MOTS + MOTSChallenge
  - [x] COCO + Cityscapes + KITTI-MOTS 
  - [x] COCO + Cityscapes + KITTI-MOTS + MOTSChallenge
 
 (c) Explore and analyze the impact of different hyperparameters
 
• Fine-tuning on KITTI-MOTS training and evaluation on KITTI-MOTS validation

• Analyze at least 3 hyperparameters you can find on the configuration file:
 
  - [x] Anchor sizes and anchor aspect ratios
  - [x] IOU overlap ratios: BG_IOU_THRESHOLD, FG_IOU_THRESHOLD
  - [x] Maximum number of region proposals. 
 
 (d) Extend Related Work and Experiments section on your paper
 
  - [x] Extend Experiments section with the experiments performed on both object detection and segmentation
  - [x] Extend Related Work section with a description of stateof-the-art object segmentation techniques


- Slides for the project: [T00-Google Slides](https://docs.google.com/presentation/d/1hpHVLfExs58Ks25nwExQLpDPJom7CD98961NIXcQ_4s/edit?usp=sharing)

- Link to the Overleaf article (non-editable): [Group00-Overleaf](https://www.overleaf.com/read/ryjfqgkckfdx)
