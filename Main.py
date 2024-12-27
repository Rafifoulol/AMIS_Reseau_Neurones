import kagglehub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Dowload les Data depuis le site Kaggle
path = kagglehub.dataset_download("thomasdubail/brain-tumors-256x256")
path += "/Data/normal/N_100.jpg"
# Affiche le chemin sur ta machine pour y accédé
print("Path to dataset files:", path)

# Affiche l'image trouver (exemple ici)
image = mpimg.imread(path)
plt.imshow(image)
plt.show()
