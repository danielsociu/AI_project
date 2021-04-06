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

# MinMaxScaler este cel mai bun
# Normalising data echivalent cu impartirea /255 (pt minmax)
#preprocessor_SVM = RobustScaler()
preprocessor_SVM = MinMaxScaler()
#preprocessor_SVM = StandardScaler()
X_train_SVM = preprocessor_SVM.fit_transform(X_train)
X_valid_SVM = preprocessor_SVM.transform(X_valid)
X_test_SVM = preprocessor_SVM.transform(X_test)

# Loading prea lung, asa ca pentru teste am folosit doar 10k
# Si pentru ca folosesc mai putine date, am decis sa folosesc train_test_split
# Evident cand am trimis l-am antrenat pe tot train-ul
# 0.73 on all data with a train split, and 0.75 on kaggle for train+valid train
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train_SVM[:10000],
    y_train[:10000],
    train_size=0.8,
    random_state=0
)
# Evident ca SVC deoarece vorbim despre un classifier!
SVC_model = SVC(
    C=1.0, # 1 pare sa produca cel mai bun scor
    kernel='rbf', # kernel-ul rbf a fost cel mai bun
    cache_size=4000,
    tol=1e-5,
    decision_function_shape="ovr", # ovr e mai bun decat ovo in cazul acesta
    class_weight='balanced',
    random_state=0
)

# Antrenare
SVC_model.fit(X_train_split, y_train_split)

# Prezicere
predictions = SVC_model.predict(X_valid_split)
score = accuracy_score(y_valid_split, predictions)

# afisare scor
confusion_matrix_SVC = confusion_matrix(
    y_valid_split,
    predictions,
    normalize='true'
)
plot_cm_SVC = sn.heatmap(confusion_matrix_SVC, annot=True, cmap=plt.cm.Reds)
plot_cm_SVC.get_figure().savefig('plot_cm_SVC_ovr_c')

print ('Score: ', score)

# Diferite train-ing uri pe care le-am incercat si scorul lor(acuratetea)
# ### Trainings done on 10k data so it's a reasonable wait time
# * SVC_model = SVC(
#     C=0.8,
#     kernel='rbf',
#     cache_size=4000,
#     tol=1e-5,
#     decision_function_shape="ovo",
#     random_state=0
# )
#     * 0.648
# * SVC_model = SVC(
#     C=0.8,
#     kernel='rbf',
#     cache_size=4000,
#     tol=1e-5,
#     decision_function_shape="ovr",
#     class_weight='balanced',
#     random_state=0
# )
#     * 0.65
# * SVC_model = SVC(
#     C=1.0,
#     kernel='rbf',
#     cache_size=4000,
#     tol=1e-5,
#     decision_function_shape="ovr",
#     class_weight='balanced',
#     random_state=0
# )
#     * 0.6605
#     * 0.73 on all data with a train split, and 0.75 on kaggle for train+valid train

##Sender SVM
##preprocessor_final_SVM = RobustScaler()
#preprocessor_final_SVM = MinMaxScaler()
##preprocessor_final_SVM = StandardScaler()
#X_train_total_final_SVM = preprocessor_final_SVM.fit_transform(X_train_total)
#X_test_final_SVM = preprocessor_final_SVM.transform(X_test)

#best_SVC_model = SVC(
#    C=1.0,
#    kernel='rbf',
#    cache_size=4000,
#    shrinking=True,
#    tol=1e-2,
#    decision_function_shape="ovr",
#    class_weight='balanced',
#    random_state=0
#)
#best_SVC_model.fit(X_train_total_final_SVM, y_train_total)
#test_answer = best_SVC_model.predict(X_test_final_SVM)
#output = pd.DataFrame({'id': X_data_test[0],
#                       'label': test_answer})
#print(output.head())
#output.to_csv('submission_SVC.cvs', index = False)

