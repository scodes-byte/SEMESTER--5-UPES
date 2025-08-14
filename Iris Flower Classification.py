# Iris Flower Classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # For decision boundary (use only first two features)
y = iris.target

# EDA
plt.figure(figsize=(6, 4))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette="Set1")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris EDA (Sepal length vs width)")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Choose model: 'knn', 'decision_tree', or 'logistic'
model_choice = 'knn'

if model_choice == 'knn':
    model = KNeighborsClassifier(n_neighbors=5)
elif model_choice == 'decision_tree':
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
else:
    model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Decision boundary plot
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set1)
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=iris.target_names[y_train], palette="Set1", edgecolor='k')
plt.title(f"Iris Decision Boundary - {model_choice}")
plt.show()
