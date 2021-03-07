import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
torch.cuda.empty_cache()



EXPERIMENT_NAME = '_pytorch_M3_architecture_30_Epochs'

# setup cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('file.png')

num_epochs = 30
batch_size = 32
#train_data_dir = '../DATASETS/MIT_split/train'
#test_data_dir = '../DATASETS/MIT_split/test'
#val_data_dir='/home/mcv/datasets/MIT_split/test'
train_data_dir='/home/mcv/datasets/MIT_split/train'
test_data_dir='/home/mcv/datasets/MIT_split/test'

# DATA TRANSFORMS
transform = transforms.Compose(
    [#transforms.ToPILImage(),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


## CREATE TEST/TRAIN/VALIDATION DATA AND DATALOADER STRUCTURES
validation_split = 0.1
train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
'''
train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [int(len(train_dataset)-int(train_dataset*validation_split)), int(len(train_dataset)*validation_split)], 
        generator=torch.Generator().manual_seed(1))
'''
train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(test_data_dir, transform=transform)

test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True)
validation_loader = test_loader

# AUX FUNCTIONS FOR LAYER MANAGEMENT
################################ 
class Print(nn.Module):
    def forward(self, x):
        print("FC1 INPUT SIZE:")
        print(x.size())
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,128*3*3)
################################


net = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3,32,3)),
    ('relu1', nn.ReLU()),
    ('maxPool1', nn.MaxPool2d(2)),
    ('conv2', nn.Conv2d(32,32,3)),
    ('relu2', nn.ReLU()),
    ('maxPool2', nn.MaxPool2d(2)),
    ('conv3', nn.Conv2d(32,64,3)),
    ('relu3', nn.ReLU()),
    ('batchNorm1', nn.BatchNorm2d(64)),
    ('conv4', nn.Conv2d(64,64,3)),
    ('relu4', nn.ReLU()),
    ('conv5', nn.Conv2d(64,256,3)),
    ('relu5', nn.ReLU()),
    ('maxPool3', nn.MaxPool2d(2)),
    ('conv6', nn.Conv2d(256,128,3)),
    ('relu6', nn.ReLU()),
    ('batchNorm2', nn.BatchNorm2d(128)),
    ('conv7', nn.Conv2d(128,128,3)),
    ('relu7', nn.ReLU()),
    ('maxPool4', nn.MaxPool2d(2)),
    ('conv8', nn.Conv2d(128,128,3)),
    ('relu8', nn.ReLU()),
    ('maxPool5', nn.MaxPool2d(2)),
    ('conv9', nn.Conv2d(128,128,3)), 
    ('relu9', nn.ReLU()),
    #('print',Print()),
    ('Flatten', Flatten()),
    #('print',Print()),
    ('fc1', nn.Linear(128*3*3, 1024)),
    ('fc2', nn.Linear(1024, 512)),
    ('fc3', nn.Linear(512, 256)),      
    ('fc4', nn.Linear(256, 8)) 
    ])) 


###### PARAMS
total_samples = len(train_dataset)
n_iterations = np.ceil(total_samples/batch_size)
print(total_samples,n_iterations)

# NN to CUDA
#net = Net()
net = net.cuda()
net.to(device)

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(net.parameters(), lr=1.0)

lr_gamma = 0.7 # Learning rate step gamma
scheduler = StepLR(optimizer, step_size=1, gamma=lr_gamma)

# Training the NN
#adding to tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('../outs/')

train_accuracy_graph = []
train_loss_graph = []
for epoch in range(num_epochs):
    running_loss = 0
    correct = 0
    loss_epoch = 0
    for i, (inputs,labels) in enumerate(train_loader):
        # forward, backward, update

        inputs = inputs.cuda()
        inputs = inputs.to(device)
        labels = labels.cuda()
        labels = labels.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).float().sum()

        
        if (i+1) % 20 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        loss_epoch = loss.item()
        running_loss += loss_epoch

    # VALIDATION
    val_total = 0
    val_correct = 0
    running_val_loss = 0
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(validation_loader):
            #print("Validation step "+str(i))
            inputs = inputs.to(device)
            labels = labels.to(device)

            val_outputs = net(inputs)
            val_outputs = val_outputs.to(device)

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (val_predicted == labels).sum().item()
            val_loss = criterion(val_outputs, labels)
            running_val_loss += val_loss.item()

    val_accuracy = (100 * val_correct / val_total)
    val_loss = running_val_loss/len(validation_loader)
    train_loss = running_loss/len(train_loader)



    


    accuracy = 100 * correct / 1881
    print(f"\n************ Accuracy = {accuracy}*******************")
    print(f"************ Loss = {train_loss:.4f}*****************\n")
    print(f"\n************ Validation Accuracy = {val_accuracy}*******************")
    print(f"************ Loss = {val_loss:.4f}*****************\n")
    scheduler.step()
    writer.add_scalar('Loss/train'+EXPERIMENT_NAME, train_loss, epoch)
    writer.add_scalar('Accuracy/train'+EXPERIMENT_NAME, accuracy, epoch)
    writer.add_scalar('Loss/validation'+EXPERIMENT_NAME, val_loss, epoch)
    writer.add_scalar('Accuracy/validation'+EXPERIMENT_NAME, val_accuracy, epoch)



print('Finished Training')


PATH = './pytorch_M3_architecture.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for (inputs,labels) in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        outputs = outputs.to(device)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 801 test images: %d %%' % (
    100 * correct / total))

with open('pytorch_M3_architecture.txt','a') as file:
                file.writelines('Accuracy of the network on the 801 test images: %d %%' % (100 * correct / total))
