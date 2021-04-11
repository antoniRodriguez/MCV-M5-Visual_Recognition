

# Week 3: Hands on Detectron -- FasterRCNN, Retina, FineTuning. 

#### Dataset
MOTSChallenge data

Dataset consist of 4 training sequences with 2862 frames and 26894 masks of pedestrians.

Data set considers Class ID:
2 for pedestrian

In addition Pedestrians as 2000, 2001, 2002 etc

KITTI-MOTS challenge 

Data consist of 12 sequences where you can find 8073 pedestrian masks and 18831 car masks. Followed 
by Validation data consist of 9 sequences 3347 pedestrians and 8068 car masks.

Data set considers two Class ID:
1 for car
2 for pedestrian

In addition Car instances are marked as 1000, 1001, 1002 etc
and Pedestrians as 2000, 2001, 2002 et


   This project presents the work realized during Module-5 where an analysis of the different Facebook's Detectron2 framework for object detection and segmentation is made using KITTI-MOTS and MOTS challenge datasets to fine-tune the models. During the project, many different configurations are used, from pre-trained models inference and evaluation to trained models on any dataset and evaluation using a different datasets. AP metric is used to evaluate the performance on cars and pedestrians. 

Inside `week3/` are the files with implementation of Faster RCNN, Retina, FineTuning script as well as supporting scripts implemented in pytorch

Tasks to be done: 
  - [x] Get familiar with MOTSChallenge and KITTI-MOTS challenges 
  - [x] Use object pre-trained object detection models in inference on KITTI-MOTS 
  - [x] Evaluate pre-trained object detection models on KITTI-MOTS 
  - [x] Train detection models on KITTI-MOTS and MOTSChallenge training sets
  - [x] Evaluate best trained model on KITTI-MOTS validation set.
  - [x] Write an Experiments section on your paper.

- Slides for the project: [T00-Google Slides](https://docs.google.com/presentation/d/19-akB_E8qloRWHGxoj8BHF7iqcElG2b4ZQSPXm6Ia4U/edit?usp=sharing)

- Link to the Overleaf article (non-editable): [Group00-Overleaf](https://www.overleaf.com/read/ryjfqgkckfdx)
