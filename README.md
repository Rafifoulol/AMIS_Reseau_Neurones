# Projet MASTER 2 AMIS : Reseau de neurone

### GUERIN Raphael et FERNANDEZ Sebastien

![](/Images/UVSQ_Logo.png)
Image prise de : https://fr.wikipedia.org/wiki/Universit%C3%A9_de_Versailles_%E2%80%93_Saint-Quentin-en-Yvelines

## Présentation du projet

Ce projet a pour objectif d'implémenter et d'entraîner un réseau de neurones à l’aide des frameworks PyTorch ou Keras, afin de résoudre un problème spécifique. Dans notre cas, nous avons choisi d'entrainer un réseau de neurones sur des images d'IRM et de CT-scan du cerveau. L’objectif est d’entraîner un réseau de neurones capable, à partir d'une image, de déterminer si elle provient d’un patient atteint d’un cancer ou non. 

Pour commencer, nous avons recherché un dataset adapté. Celui que nous avons utilisé provient de [Kaggle](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri). Ce dataset contient 9620 images de taille différentes. 

![Images d'un Cerveau Sain](/Images/N_3.jpg)

## Problématique

Est-il possible d'entrainer suffissament un réseau de neurones pour qu’il fournisse un diagnostic avec un taux de réussite proche de 100 % ? Et, quel type de réseau de neurones est le plus performant pour cette tâche ?

Dans le projet nous avonc donc implémenté : 
- Un **MLP (Multilayer perceptron)** à 4 couches
- Un **CNN (Convolutional neural network)**

## MLP

### Dataset

Pour le MLP étant un réseau de neurones prenant en entré, la taille des données qui l'entrainerons, il fallait un dataset un peu plus particulier que le dataset choisit. Nous avons remarqué que le dataset possèdait un sous [dataset](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256) contenant 3096 images d'IRM, de tailles 256x256. Nous avons adapté le dataset afin de réduire encore la taille des images et simplifier l’entraînement. Toutes les images ont été redimensionnées à 128x128 pixels, et bien qu’elles possèdent initialement 3 canaux de couleurs (rouge, vert, bleu), nous avons conservé uniquement les niveaux de gris. Cela permet de réduire la taille des données en entrée, tout en préservant l’essentiel des informations nécessaires pour l’analyse des images.

Ainsi, chaque image redimensionnée est convertie en un vecteur plat de taille 128×128=16384 (au lieu de 3x256x256=131072), correspondant à la taille de la couche d’entrée du réseau.

Le dataset contient des images réparties en quatre catégories :
- **Glioma**
- **Meningioma**
- **Normal (absence de cancer)**
- **Pituitary**


### Modèle

Le MLP a été défini avec l’architecture suivante :
- Une couche d’entrée contenant 16384 neurones (taille des images converties en vecteurs).
    - Deux couches cachées :
        - La première avec 4096 neurones.
        - La seconde avec 512 neurones.
- Une couche de sortie composée de 4 neurones.

Les connexions entre les couches sont assurées par des fonctions linéaires (Linear layers) suivies de la fonction d’activation ReLU (Rectified Linear Unit) pour introduire des non-linéarités.

Le modèle est défini comme suit :
``` python
class MLP(nn.Module) :
  def __init__(self, input_size=16384, output_size=4, layers=[4096, 512]):
    super().__init__()
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(input_size, layers[0]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Linear(layers[0], layers[1]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Linear(layers[1], output_size))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
```

### Entraînement

Pour entraîner ce modèle, nous avons suivi les étapes suivantes :

1. **Prétraitement des données :**
   - Redimensionnement des images à 128x128 pixels.
   - Conversion des images en niveaux de gris (un seul canal).
   - Mise à plat des images en vecteurs de taille 16384.
   - Normalisation des valeurs des pixels dans l’intervalle [0,1].

2. **Configuration de l’entraînement :**
   - Optimiseur : Adam, avec un taux d’apprentissage initial fixé à 0.001.
   - Fonction de perte : Cross-Entropy Loss, adaptée à la classification en 4 catégories.
   - Taille des batchs : 129 images par batch (car 129x24 = 3096 notre nombre d'images).
   - Nombre d’époques : 5, 10, 15 (pour comparer).

3. **A faire :**
   - A compléter
   - Ajouter les graphes

### Forces et Faiblesses

**Forces :**
- Le modèle est léger et rapide à entraîner.
- Prétraitement des données optimisé pour la mémoire.

**Faiblesses :**
- Performance limitée à cause de l’architecture MLP.
- Le modèle ne prend pas en compte les relations spatiales entre les pixels.


### Améliorations possibles

Pour améliorer ce modèle, plusieurs pistes sont envisageables :

- Augmentation des données : Appliquer des transformations comme des rotations, des zooms ou des inversions pourrait enrichir le dataset et améliorer la généralisation.
- Réduction encore plus marquée de la dimensionnalité : Explorer des approches comme les autoencodeurs pour compresser les images avant leur entrée dans le MLP.
- Exploration de la topologie des couches : Ajouter ou supprimer des couches cachées ou expérimenter avec un nombre différent de neurones.
- Utilisation de prétraitements avancés : Introduire des méthodes comme la détection de contours ou des transformations spectrales pourrait améliorer les performances.

## CNN


