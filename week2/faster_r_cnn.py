import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
import os, json, cv2, random
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
from detectron2.utils.logger import setup_logger
setup_logger()

num_epochs = 150
batch_size = 8
#train_data_dir = '../DATASETS/MIT_split/train'
#test_data_dir = '../DATASETS/MIT_split/test'
#val_data_dir='/home/mcv/datasets/MIT_split/test'
# train_data_dir='/home/mcv/datasets/MIT_split/train'
train_data_dir='/home/mcv/datasets/MIT_split/train'
test_data_dir='/home/mcv/datasets/MIT_split/test'

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    good_image = np.transpose(npimg, (1, 2, 0))*255
    cv2.imwrite('file_adam.png', good_image)
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.savefig('file.png')
    return good_image

# DATA TRANSFORMS
transform = transforms.Compose(
    [#transforms.ToPILImage(),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


## CREATE TEST/TRAIN/VALIDATION DATA AND DATALOADER STRUCTURES
validation_split = 0.1
train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(test_data_dir, transform=transform)

test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True)
validation_loader = test_loader



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# setup cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()
model.to(device)

# For inference
dataiter = iter(train_loader)
images, labels = dataiter.next()

inputs = images.cuda()
inputs = inputs.to(device)
model.eval()
outputs = model(inputs)

print(outputs[0])
# print(outputs[0]["instances"].pred_classes)
# print(outputs[0]["instances"].pred_boxes)
from test_boxes import draw_box_save
for idx,item in enumerate(outputs):
    # idx = 2
    good_image = imshow(images[idx])
    filename=f"img_{19+idx}"
    draw_box_save(outputs[idx]['boxes'], outputs[idx]['labels'], outputs[idx]['scores'],filename, 'file_adam.png')

# v = Visualizer(im[:, :, ::-1], scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imwrite("file.jpg",out.get_image()[:, :, ::-1])
