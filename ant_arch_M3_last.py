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

# TODO: 
# Adapt network to the latest version we had
# Training accuracy - plot and check 
# Optimizer - Adadelta
# Learning rate

EXPERIMENT_NAME = 'keras_to_pytorch_20Epochs'
# setup cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('file.png')


''' TENSORFLOW MODEL
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.Conv2D(256,(3,3),activation='relu'))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

'''
num_epochs = 20
batch_size = 1
train_data_dir = '../DATASETS/MIT_split/train'
test_data_dir = '../DATASETS/MIT_split/test'
train_data_dir='/home/mcv/datasets/MIT_split/train'
#val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'

# DATA AUGMENTATION
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
'''
validation_loader = DataLoader(validation_subset,
                          batch_size = batch_size,
                          shuffle=True)
'''
test_dataset = torchvision.datasets.ImageFolder(test_data_dir, transform=transform)

test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True)
validation_loader = test_loader

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        print("Defining architecture...")
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.batchnorm = nn.BatchNorm2d(64)
        #Activation Layer
        self.conv3 = nn.Conv2d(64, 64, 3)
        #self.batchnorm2 = nn.BatchNorm2d(64)
        # Activation Layer
        self.globalAvgPooling = nn.AvgPool2d(1, stride=2)
        #### get shape
        self.fc1 = nn.Linear(64 * 125 * 125, 2048)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048, 8)
        print("Architecture defined")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm(x)

        # Activation layer #1
        x = F.relu(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm(x)

        # Activation layer #2
        x = F.relu(x)
        x = self.globalAvgPooling(x)
        x = x.view(-1, 64 * 125 * 125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''

'''
model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)))x
    model.add(layers.Conv2D(64, (3,3),activation='relu'))x
    model.add(layers.BatchNormalization())x
    model.add(layers.Activation('relu'))x
    model.add(layers.Conv2D(64, (3,3),activation='relu'))x
    model.add(layers.BatchNormalization())x
    model.add(layers.Activation('relu'))x
    model.add(layers.Conv2D(64, (3,3),activation='relu'))x
    model.add(layers.BatchNormalization())x
    model.add(layers.Activation('relu'))x
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.GaussianNoise(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='softmax'))
'''
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

    ('fc1', nn.Linear(128 * 125 * 125, 1024)),
    ('fc2', nn.Linear(1024, 512)),
    ('fc3', nn.Linear(512, 256)),      
    ('fc4', nn.Linear(256, 8))
]))


###### PARAMS
total_samples = len(train_dataset)
n_iterations = np.ceil(total_samples/batch_size)
print(total_samples,n_iterations)

# NN creation
#net = Net()
net = net.cuda()
net.to(device)

print("Created network")
# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
#optimizer = optim.Adamax(net.parameters(), lr=0.01)
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
        print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

        inputs = inputs.cuda()
        inputs = inputs.to(device)
        labels = labels.cuda()
        labels = labels.to(device)

        # Forward pass
        print("Begin forward...")
        outputs = net(inputs)
        print("Ended forward...")
        print("Start loss...")
        loss = criterion(outputs, labels)
        print("Ended loss...")

        # Backward and optimize
        optimizer.zero_grad()
        print("Begin loss backward...")
        loss.backward()
        print("Ended loss backward...")
        print("Begin optimizer.step...")
        optimizer.step()
        print("Ended optimizer.step...")
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
            print("Validation step "+str(i))
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
    print(f"\n************Accuracy = {accuracy}*******************")
    print(f"************Loss = {train_loss:.4f}*****************\n")
    scheduler.step()
    writer.add_scalar('Loss/train'+EXPERIMENT_NAME, train_loss, epoch)
    writer.add_scalar('Accuracy/train'+EXPERIMENT_NAME, accuracy, epoch)
    writer.add_scalar('Loss/validation'+EXPERIMENT_NAME, val_loss, epoch)
    writer.add_scalar('Accuracy/validation'+EXPERIMENT_NAME, val_accuracy, epoch)



print('Finished Training')


PATH = './adadelta-augmentation.pth'
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

with open('Adam-Adamax-Augmentation.txt','a') as file:
                file.writelines('Accuracy of the network on the 801 test images: %d %%' % (100 * correct / total))
