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
    
    


def process_data(dataloader, network):
    """
    Pass a dataloder into a network.
    Return targets and predictions.
    """

    all_targets = []
    all_predictions = []

    for b in dataloader:

        # samples and targets
        samples = reshape_batch(b[0])
        targets = b[1]
        all_targets.extend(targets)

        # forward pass
        outputs = network.forward(samples)
        predictions = torch.argmax(outputs, dim=0)
        all_predictions.extend(predictions)
    
    return all_targets, all_predictions

num_epochs = 20
batchsize = 100
lr = 0.001

EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "/home/rafifou/.cache/kagglehub/datasets/thomasdubail/brain-tumors-256x256/versions/1/Data"
TEST_DATA_PATH = "/home/rafifou/.cache/kagglehub/datasets/thomasdubail/brain-tumors-256x256/versions/1/DataTest"


train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=ToTensor())
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=ToTensor())

print(len(train_data))
print(len(test_data))


def reshape_batch(b) :
    # Step 1: Remove the channel dimension (use only one channel, e.g., the first channel)
    b = b[:, 1, :, :]
    b = torch.squeeze(b)
    b = torch.flatten(b, 1, 2)
    b = b.transpose(0, 1)

    return b

# La je récupére la 47eme photo de mes Data. Je sépare l'image et le Label
sample = train_data[47][0]
target = train_data[47][1]

# Je ne prend qu'une des couleurs car mon sample.torch.Size(3, 256, 256)
# Donc je ne veux qu'une des trois couleur pour etre de la forme (1, 256, 256)
# Je le shape en 256x256 pour l'afficher sur matplot
sample_shaped = sample[0]
sample_shaped = sample_shaped.view(256,256)
#plt.imshow(sample_shaped)
#plt.show()

# Maintenant je le flatten
sample_1d = sample_shaped.flatten()
#print(sample_1d.shape) # Taille 65536

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

for b in train_data_loader :
    break
#print(b[0].shape, b[1].shape) # torch.Size([100, 3, 256, 256]) torch.Size([100])

b_reshaped = reshape_batch(b[0])
#print(b_reshaped.shape)

mlp = MLP ([65536, 4096, 128, 4])

all_targets, all_predictions = process_data(train_data_loader, mlp)
print("Train results\n")
print(classification_report(all_targets, all_predictions))


