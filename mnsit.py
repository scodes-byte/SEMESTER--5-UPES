import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Download dataset from Kaggle
os.system('kaggle datasets download -d oddrationale/mnist-in-csv')

# 2️⃣ Extract the ZIP file (Windows safe)
with zipfile.ZipFile('mnist-in-csv.zip', 'r') as zip_ref:
    zip_ref.extractall()

# 3️⃣ Load CSV
train_df = pd.read_csv('mnist_train.csv')

# 4️⃣ Separate labels and pixel data
y_train = train_df['label']
X_train = train_df.drop('label', axis=1)

# 5️⃣ Plot first 10 images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train.iloc[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
