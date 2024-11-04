import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import Network

data = pd.read_csv("train.csv")

label = data['label']
img = data.drop('label', axis=1)
img_tensor = np.array(img).reshape(-1, 1, 32, 32)

dataset = torch.utils.data.TensorDataset(torch.tensor(img_tensor, dtype = torch.float).to("cuda"), torch.tensor(label, dtype = torch.long).to("cuda"))
generator = torch.Generator().manual_seed(17)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator)

training_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
testing_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
        

class DenseNetwork(Network):
    def __init__(self, training_loader, validation_loader, alpha=0, dropout_prob=0):
        super().__init__(training_loader, validation_loader, alpha, dropout_prob)
        self.fc1 = nn.Linear(32*32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.drop = nn.Dropout(p=dropout_prob)
        self.fco = nn.Linear(16, 10)

    # dopredny chod
    def forward(self, x):
        x = x.flatten(start_dim = 1)
        x = F.relu(self.fc1(x))
        if self.dropout_prob != 0:
            x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fco(x)
        return x
    


model_dense = DenseNetwork(training_loader, validation_loader).to("cuda")
model_dense.set_parameters(torch.nn.CrossEntropyLoss(), torch.optim.Adam(model_dense.parameters()))
# 50 epoch, early stopping po 10 epochách
model_dense.train(50, 10)


class ConvNetwork(Network):
    def __init__(self, training_loader, validation_loader, alpha=0, dropout_prob=0):
        super().__init__(training_loader, validation_loader, alpha, dropout_prob)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fco = nn.Linear(576, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        if self.dropout_prob:
            x = self.conv1_drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(start_dim = 1)
        x = self.fco(x)
        return x
    

model_conv = ConvNetwork(training_loader, validation_loader).to("cuda")
model_conv.set_parameters(torch.nn.CrossEntropyLoss(), torch.optim.Adam(model_conv.parameters()))
model_conv.train(50, 10)