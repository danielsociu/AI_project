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

# Again si aici minmaxscaler este cel mai bun, cel putin cand tratez cazuri cu aceiasi parametri
# Normalising data echivalent cu impartirea /255 pt minmax
#preprocessor_LDA = RobustScaler()
preprocessor_LDA = MinMaxScaler()
#preprocessor_LDA = StandardScaler()
X_train_LDA = preprocessor_LDA.fit_transform(X_train)
X_valid_LDA = preprocessor_LDA.transform(X_valid)
X_test_LDA = preprocessor_LDA.transform(X_test)

# Default gets 0.588
# lsqr with shrinkage auto -> 0.596
# Linear Discriminant Analasys model
linear_discriminant_model = LinearDiscriminantAnalysis(
    solver='lsqr',
    shrinkage='auto'
)
linear_discriminant_model.fit(X_train_LDA, y_train)
score = linear_discriminant_model.score(X_valid_LDA, y_valid)

predictions = linear_discriminant_model.predict(X_valid_LDA)
score = accuracy_score(y_valid, predictions)

confusion_matrix_LDA = confusion_matrix(
    y_valid,
    predictns,
    normalize='true'
)
plot_cm_LDA = sn.heatmap(confusion_matrix_LDA, annot=True, cmap=plt.cm.Reds)
plot_cm_LDA.get_figure().savefig('plot_cm_LDA_lsqr_a')

print ('Score: ', score)

# Incercare eigen, obtine mai bine
#     - 0.604

linear_discriminant_model = LinearDiscriminantAnalysis(
    solver='eigen',
    covariance_estimator=covariance.ShrunkCovariance()
)
linear_discriminant_model.fit(X_train_LDA, y_train)
score = linear_discriminant_model.score(X_valid_LDA, y_valid)
# print (score)
predictions = linear_discriminant_model.predict(X_valid_LDA)
score = accuracy_score(y_valid, predictions)

# Afisare matrice confuzie si scor
confusion_matrix_LDA = confusion_matrix(
    y_valid,
    predictions,
    normalize='true'
)
plot_cm_LDA = sn.heatmap(confusion_matrix_LDA, annot=True, cmap=plt.cm.Reds)
plot_cm_LDA.get_figure().savefig('plot_cm_LDA_eigen')

print ('Score: ', score)


# cel mai bun model gasit de mine:
#     * 0.605
linear_discriminant_model = LinearDiscriminantAnalysis(
    solver='lsqr',
    covariance_estimator=covariance.ShrunkCovariance()
)
linear_discriminant_model.fit(X_train_LDA, y_train)
score = linear_discriminant_model.score(X_valid_LDA, y_valid)
# print (score)
predictions = linear_discriminant_model.predict(X_valid_LDA)
score = accuracy_score(y_valid, predictions)

confusion_matrix_LDA = confusion_matrix(
    y_valid,
    predictions,
    normalize='true'
)
plot_cm_LDA = sn.heatmap(confusion_matrix_LDA, annot=True, cmap=plt.cm.Reds)
plot_cm_LDA.get_figure().savefig('plot_cm_LDA_lsqr_cov')

print ('Score: ', score)

# * linear_discriminant_model = LinearDiscriminantAnalysis()
#     * 0.588
# * linear_discriminant_model = LinearDiscriminantAnalysis(
#     solver='lsqr',
#     shrinkage='auto'
# )
#     - 0.596
# * linear_discriminant_model = LinearDiscriminantAnalysis(
#     solver='eigen',
#     covariance_estimator=covariance.ShrunkCovariance()
# )
#     * 0.604
# * linear_discriminant_model = LinearDiscriminantAnalysis(
#     solver='lsqr',
#     covariance_estimator=covariance.ShrunkCovariance()
# )
#     * 0.605


# Submission
# #preprocessor_final_LDA = RobustScaler()
# preprocessor_final_LDA = MinMaxScaler()
# #preprocessor_final_LDA = StandardScaler()
# X_train_total_final_LDA = preprocessor_final_LDA.fit_transform(X_train_total)
# X_test_final_LDA = preprocessor_final_LDA.transform(X_test)
# 
# best_linear_discriminant_model = LinearDiscriminantAnalysis(
#     solver='lsqr',
#     covariance_estimator=covariance.ShrunkCovariance()
# )
# 
# best_linear_discriminant_model.fit(X_train_total_final_LDA, y_train_total)
# test_answer = best_linear_discriminant_model.predict(X_test_final_LDA)
# output = pd.DataFrame({'id': X_data_test[0],
#                        'label': test_answer})
# print(output.head())
# output.to_csv('submission_LDA.cvs', index = False)
