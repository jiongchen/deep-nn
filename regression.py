#!/usr/bin/python

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from mat_io import read_dense_matrix
    
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(8, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ac1(out)
        out = self.fc2(out)
        out = self.ac2(out)
        out = self.fc3(out)
        return out

x_row, x_col, x_dat = read_dense_matrix('./train.dat')
y_row, y_col, y_dat = read_dense_matrix('./label.dat')
assert x_col == y_col
print("input and output dim (%d, %d)" % (x_row, y_row))
print("sample num %d" % x_col)

model = Net(x_row, y_row)
model.double()
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 15000

x_train = np.transpose(x_dat)
y_train = np.transpose(y_dat)

for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(x_train).cuda())
    targets = Variable(torch.from_numpy(y_train).cuda())

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1)%5 == 0:
        print ('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.data[0]))

predicted = model(Variable(torch.from_numpy(x_train)).cuda()).cpu().data.numpy()
plt.plot(y_train, label='groud_truth')
plt.plot(predicted, label='fitted result')
plt.legend()
plt.show()
