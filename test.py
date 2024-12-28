import kagglehub
import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision.transforms import ToTensor
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Voici le path pour accéder aux images récupéré sur Kaggle (le 2eme dossier je l'ai créer à partir du premier pour les tests)
TRAIN_DATA_PATH = "/home/rafifou/.cache/kagglehub/datasets/thomasdubail/brain-tumors-256x256/versions/1/Data"
TEST_DATA_PATH = "/home/rafifou/.cache/kagglehub/datasets/thomasdubail/brain-tumors-256x256/versions/1/DataTest"

# On va load les data
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=ToTensor())
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=ToTensor())

print(len(train_data))
print(len(test_data))

# Je prend un échantillon (100 et 50 est un hasard je peux prendre plus loin)
train_sample, train_label = train_data[100]
test_sample, test_label = test_data[50] 

# Je vérifie les données de mon échantillon
print(f"Sample shape: {train_sample.shape}")
print(f"Class name: {train_data.classes[train_label]}")

# Convert tensors to numpy arrays for visualization
train_sample_np = train_sample.permute(1, 2, 0).numpy()
test_sample_np = test_sample.permute(1, 2, 0).numpy()

# J'affiche l'un à coté de l'autre les 2 images avec matplot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(train_sample_np)
plt.title(f"Train: {train_data.classes[train_label]} (Label: {train_label})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(test_sample_np)
plt.title(f"Test: {test_data.classes[test_label]} (Label: {test_label})")
plt.axis("off")

plt.tight_layout()
plt.show()


# Avec ces lignes on peut changer les couleurs (ici les gris)
sample = train_data[47][0]
target = train_data[47][1]

grayscale_sample = sample[0]  # Select the first channel (red)
print(grayscale_sample.shape)  # torch.Size([256, 256])

plt.imshow(grayscale_sample, cmap="Greys")
plt.axis("off")
plt.show()