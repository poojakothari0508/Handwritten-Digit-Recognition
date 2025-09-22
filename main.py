import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.images
y = digits.target

n_samples = len(X)
X_flat = X.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

_, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, prediction in zip(axes.flatten(), X_test.reshape(-1, 8, 8), y_pred):
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f"Pred: {prediction}")
    ax.axis('off')
plt.show()
