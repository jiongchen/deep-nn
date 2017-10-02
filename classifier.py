#!/usr/bin/python

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn.functional as F
import sys, os
from PIL import Image
import numpy as np


transform = transforms.Compose([transforms.ToTensor()])

class RawImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.train_data = []
        self.train_labels = []

        for img_file in os.listdir(img_dir):
            print('reading image %s' % img_file)

            # image (H, W, C)
            im = np.array(Image.open(os.path.join(img_dir, img_file)))
            print(im.shape)
            self.train_data.append(im)

            # get label
            la = int(img_file[-6:-4])
            print(la)
            self.train_labels.append(la)

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.train_data)
    

batch_size = 500
train_dataset = RawImageDataset('./dat/cifar', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
                                                                        

learning_rate = 0.001
num_epochs = 500

cnn = CNN(num_classes=10)
cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # for a batch
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, loss.data[0]))

torch.save(cnn.state_dict(), 'model.pkl')
