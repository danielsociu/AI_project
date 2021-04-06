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

# Naive Bayes
# convertim in valorile in buckets, asemanator cu normalizarea
def values_to_bins(data, bins):
    bins_data = np.digitize(data, bins)
    return bins_data - 1
bins_scores = []
naive_bayes_model = MultinomialNB(alpha=0.2)
# We've seen bins = 12 is the best for this case so we'll be working with 12 bins
# for num_bins in range(2,50,2):
#     bins = np.linspace(start=0, stop = 255, num=num_bins)
#     train = values_to_bins(X_train, bins)
#     valid = values_to_bins(X_valid, bins)
#     naive_bayes_model.fit(train, y_train)
#     score = naive_bayes_model.score(valid, y_valid)
#     print ('bins: ' + str(num_bins) + ':',  score)
#     bins_scores.append({"bins": num_bins, "score": score})
# for alpha in np.arange(0.1,1.0,0.1):
#     naive_bayes_model = MultinomialNB(alpha=alpha)
#     bins = np.linspace(start=0, stop = 255, num=12)
#     # apparently data is too big for alpha to matter? -> dividing by 12 looks the best
#     train = values_to_bins(X_train, bins)/12 # small normalisation
#     valid = values_to_bins(X_valid, bins)/12
#     naive_bayes_model.fit(train, y_train)
#     score = naive_bayes_model.score(valid, y_valid)
#     print ('Alpha: ' + str(alpha) + ': ', score)

# prelucrare date, le transformam pixelii in buckets asa cum am facut la laborator
bins = np.linspace(start=0, stop = 255, num=12)
train = values_to_bins(X_train, bins)
valid = values_to_bins(X_valid, bins)

# Multinomial:  0.392
# Antrenez pe multinomial, alpha nu afecteaza cu nimic
naive_bayes_model = MultinomialNB(alpha=0.2)
# Antrenare
naive_bayes_model.fit(train, y_train)

# Prezicere
predictions = naive_bayes_model.predict(valid)
score = accuracy_score(y_valid, predictions)

# Afisez confusion matrix pentru cazul NBM si acuratetea
confusion_matrix_NB = confusion_matrix(
    y_valid,
    predictions,
    normalize='true'
)
plot_cm_NBM = sn.heatmap(confusion_matrix_NB, annot=True, cmap=plt.cm.Reds)
plot_cm_NBM.get_figure().savefig('plot_cm_NBM')

print ('Multinomial: ', score)

# Antrenez pe bernoulli
# Bernoulli:  0.3214
bernoulli_naive_bayes = BernoulliNB(binarize=0.2)
bernoulli_naive_bayes.fit(train, y_train)

predictions = bernoulli_naive_bayes.predict(valid)
score = accuracy_score(y_valid, predictions)

confusion_matrix_NBB = confusion_matrix(
    y_valid,
    predictions,
    normalize='true'
)
plot_cm_NBB = sn.heatmap(confusion_matrix_NBB, annot=True, cmap=plt.cm.Reds)
plot_cm_NBB.get_figure().savefig('plot_cm_NBB')

print ('Bernoulli: ', score)

# Antrenez pe tipul gaussian:
# Gaussian:  0.3996
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(train, y_train)

predictions = gaussian_naive_bayes.predict(valid)
score = accuracy_score(y_valid, predictions)

confusion_matrix_NBG = confusion_matrix(
    y_valid,
    predictions,
    normalize='true'
)
plot_cm_NBG = sn.heatmap(confusion_matrix_NBG, annot=True, cmap=plt.cm.Reds)
plot_cm_NBG.get_figure().savefig('plot_cm_NBG')

print ('Gaussian: ', score)

# Multinomial:  0.392
# Bernoulli:  0.3214
# Gaussian:  0.3996
# the best is gaussianNB one 0.3996


