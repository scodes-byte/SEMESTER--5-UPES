import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
train = pd.read_csv("fashion_mnist_data/fashion-mnist_train.csv")

# Separate labels and pixel values
labels = train['label']
images = train.drop('label', axis=1).values

# Reshape first 9 images into 28x28
images = images.reshape(-1, 28, 28)

# Class names for labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Plot first 9 images
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(class_names[labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
