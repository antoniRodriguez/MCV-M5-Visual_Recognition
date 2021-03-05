from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim



print("STARTING...")
# setup cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BOARD_PATH = '/home/group00/working/week1/outs/'
EXPERIMENT_NAME = f'_training'

train_data_dir='/home/mcv/datasets/MIT_split/train'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=150
validation_samples=807

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(test_data_dir, transform=transform)

test_loader = DataLoader(test_dataset,
                          batch_size = batch_size,
                          shuffle=True)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('file.png')


# get some random training images

#dataiter = iter(train_loader)
#images, labels = dataiter.next()


# show images
#imshow(torchvision.utils.make_grid(images))

# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F




''' torch's conv2
nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], 
    stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, 
    dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, 
    padding_mode: str = 'zeros')

'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        print("Defining architecture...")
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 256, 3)
        self.conv6 = nn.Conv2d(256, 128, 3)

        self.conv7 = nn.Conv2d(128, 128, 3)
        self.conv8 = nn.Conv2d(128, 128, 3)
        self.conv9 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 25 * 25, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 8)
        print("Architecture defined")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(F.relu(self.conv9(x)))
        #x = F.relu(self.conv9(x))
        #print("CHECKING TENSOR SIZE")
        #print(x.size())
    
        x = x.view(-1, 128 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

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

print("Creating net...")
net = Net()
net = net.cuda()
net =net.to(device)
print("Network created!")

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(net.parameters(), lr=0.01)

dataiter = iter(train_loader)
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    print(f"{epoch} Epoch")
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs=inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        #print("before optimizer")
        optimizer.zero_grad()

        # forward + backward + optimize
        #print("before outputs")
        outputs = net(inputs)
        outputs = outputs.cuda()
        outputs = outputs.to(device)
        #print("before loss")
        loss = criterion(outputs, labels)
        #print("before backward")
        loss.backward()
        #print("before optimizer step")
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            with open('output.txt','a') as file:
                file.writelines('[%d, %5d] loss: %.3f\n' % (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0


print('Finished Training')

PATH = './Adadelta_fix.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

net = net.cuda()
net =net.to(device)

#what the network thinks of all dataset
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        outputs = outputs.cuda()
        outputs = outputs.to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 801 test images: %d %%' % (
    100 * correct / total))

with open('Adadelta_fix.txt','a') as file:
                file.writelines('Accuracy of the network on the 801 test images: %d %%' % (
    100 * correct / total))
