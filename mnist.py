import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from skimage.feature import hog
from mlxtend.data import loadlocal_mnist

images, labels = loadlocal_mnist(
    images_path='images/mnist-dataset/train-images-idx3-ubyte',
    labels_path='images/mnist-dataset/train-labels-idx1-ubyte'
)

num_images_to_process = 500
hog_features = []
for i in range(num_images_to_process):
    image = images[i]
    fd, hog_image = hog(image.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualize=True)
    hog_features.append(fd)

hog_features = np.array(hog_features)

loo = LeaveOneOut()
y_true = []
y_pred = []

svm_classifier = SVC(kernel='linear')

for train_index, test_index in loo.split(hog_features):
    X_train, X_test = hog_features[train_index], hog_features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    svm_classifier.fit(X_train, y_train)
    prediction = svm_classifier.predict(X_test)
    
    y_true.append(y_test[0])
    y_pred.append(prediction[0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_true, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Confusion Matrix:\n", conf_matrix)
