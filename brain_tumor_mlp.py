import kagglehub
import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision.transforms import ToTensor
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class MLP(object):

    def __init__(self, sizes):
        """constructor"""
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [torch.normal(0, 1, size=(n, 1)) 
                       for n in sizes[1:]] # no bias for 1st layer
        self.weights = [torch.normal(0, 1, size=(n2, n1)) 
                        for n1, n2 in zip(sizes[:-1], sizes[1:])]
        
    def forward(self, X):
        """forward pass"""
        
        if X.shape[0] != self.sizes[0]:
            raise ValueError("incorrect input dimension")
        
        for W, b in zip(self.weights, self.biases):
            X = torch.tanh(torch.mm(W, X) + b)
            
        return X
    
    def forward_penultimate(self, X):
        """forward pass until penultimate layer"""
        
        if X.shape[0] != self.sizes[0]:
            raise ValueError("incorrect input dimension")
        
        for W, b in zip(self.weights[:-1], self.biases[:1]):
            X = torch.tanh(torch.mm(W, X) + b)
            
        return X
    

num_epochs = 20
batchsize = 100
lr = 0.001

EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "/home/rafifou/.cache/kagglehub/datasets/thomasdubail/brain-tumors-256x256/versions/1/Data"
TEST_DATA_PATH = "/home/rafifou/.cache/kagglehub/datasets/thomasdubail/brain-tumors-256x256/versions/1/DataTest"


train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=ToTensor())
#train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=ToTensor())
#test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

print(len(train_data))
print(len(test_data))


