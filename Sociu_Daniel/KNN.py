import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import covariance
from skimage.io import imread
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, accuracy_score, confusion_matrix

# Aceste plot configurations sunt pentru a face ca graficele sa arate mai bine.
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


# Citire date intrare ca read_csv (numele fisierelor)
X_data_train = pd.read_csv("data/train.txt", header = None)
X_data_validation = pd.read_csv("data/validation.txt", header = None)
X_data_test = pd.read_csv("data/test.txt", header = None)

# Functie care obtine fiecare imagine in parte
def get_images(path, data):
    data_images =[]
    for i in range(data.shape[0]):
        img = imread(path + data[0][i])
        img = np.asarray(img)
        data_images.append(img)
    return np.array(data_images)

# Obtinerea imaginilor
X_train_vanilla = get_images("data/train/", X_data_train)
X_validation_vanilla = get_images("data/validation/", X_data_validation)
X_test_vanilla = get_images("data/test/", X_data_test)

# Aici schimb format-ul imaginilor, le transform intr-un vector de 1024x1 in loc de 32x32x1
X_train = X_train_vanilla.flatten().reshape(X_train_vanilla.shape[0], np.prod(X_train_vanilla.shape[1:]))
y_train = X_data_train[1]
X_valid = X_validation_vanilla.flatten().reshape(X_validation_vanilla.shape[0], np.prod(X_validation_vanilla.shape[1:]))
y_valid = X_data_validation[1]
X_test = X_test_vanilla.flatten().reshape(X_validation_vanilla.shape[0], np.prod(X_validation_vanilla.shape[1:]))

# scaler = Normalizer('max').fit(X_train)
# scaler.transform(X_train)
# scaler.transform(X_valid)
#X_train, X_valid, y_train, y_valid = train_test_split(X_flattened, y, train_size=0.7, random_state=0)
# i_size = (32,32)
# X_train

# In unele locuri am antrenat pe toate datele inainte de a trimite sursa
X_train_total = np.concatenate((X_train, X_valid))
y_train_total = np.concatenate((y_train, y_valid))

# MinMaxScaler aparent e cel mai bun scaller
#preprocessor_KNN = RobustScaler()
preprocessor_KNN = MinMaxScaler()
#preprocessor_KNN = StandardScaler()
X_train_KNN = preprocessor_KNN.fit_transform(X_train)
X_valid_KNN = preprocessor_KNN.transform(X_valid)
X_test_KNN = preprocessor_KNN.transform(X_test)

# KNN model 0.4916
# knn_scores =[]
# for i in range(5,15):
#     KNN_model = KNeighborsClassifier(i)
#     KNN_model.fit(X_train_KNN, y_train)
#     score = KNN_model.score(X_valid_KNN, y_valid)
#     print (i, score)
#     knn_scores.append({"neighbors": i, 'score': score})

# We got best score on 9 neighbors

# Apparently the algorithm doesn't change anything so auto is better in this case(ball_tree takes years)
# KNN_model = KNeighborsClassifier(
#     n_neighbors=9,
#     algorithm='ball_tree',
#     n_jobs=-1
# )
# KNN_model.fit(X_train_KNN, y_train)
# score = KNN_model.score(X_valid_KNN, y_valid)
# print ('ball_tree', score)
#
# KNN_model = KNeighborsClassifier(
#     n_neighbors=9,
#     algorithm='kd_tree',
#     n_jobs=-1
# )
# KNN_model.fit(X_train_KNN, y_train)
# score = KNN_model.score(X_valid_KNN, y_valid)
# print ('kd_tree: ', score)
#
# KNN_model = KNeighborsClassifier(
#     n_neighbors=9,
#     algorithm='brute',
#     n_jobs=-1
# )
# KNN_model.fit(X_train_KNN, y_train)
# score = KNN_model.score(X_valid_KNN, y_valid)
# print ('Brute: ', score)

# Deci in final default-ul cu 9 neighbors este cel mai bun (si p = 1, adica manhattan distance)
KNN_model = KNeighborsClassifier(
    n_neighbors=9,
    n_jobs=-1,
    p=1
)
# antrenam modelul
KNN_model.fit(X_train_KNN, y_train)
# score = KNN_model.score(X_valid_KNN, y_valid)
# Afisam scorul si matricea confuzie
predictions = KNN_model.predict(X_valid_KNN)
score = accuracy_score(y_valid, predictions)

confusion_matrix_KNN = confusion_matrix(
    y_valid,
    predictions,
    normalize='true'
)
plot_cm = sn.heatmap(confusion_matrix_KNN, annot=True, cmap=plt.cm.Reds)
plot_cm.get_figure().savefig('plot_cm_KNN')

print ('Score: ', score)

# We got 0.503 with l1 (better than l2, 0.4916)
#   KNN_model = KNeighborsClassifier(
#       n_neighbors=9,
#       n_jobs=-1,
#       p=1
#   )

