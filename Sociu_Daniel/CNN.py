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

# Din nou min max scaler pare sa se descurce cel mai bine
#preprocessor_CNN = RobustScaler()
preprocessor_CNN = MinMaxScaler()
#preprocessor_CNN = StandardScaler()
# transform imagnile inapoin in matrici (erau sub forma de vector)
X_train_CNN = preprocessor_CNN.fit_transform(X_train).reshape((X_train.shape[0],32 ,32, 1)) 
X_valid_CNN = preprocessor_CNN.transform(X_valid).reshape((X_valid.shape[0], 32,32, 1))
X_test_CNN = preprocessor_CNN.transform(X_test)

# Aici trebuie sa facem clasele in categorical (adica are 1 la clasa respectiva).
y_train_CNN = keras.utils.to_categorical(y_train, 9)
y_valid_CNN = keras.utils.to_categorical(y_valid, 9)

# Depinzand de parametrii si duratie am scazut/crescut numarul de elemente folosite in testing. Am antrenat pe toate datele de antrenare
X_test_CNN = X_valid_CNN[3000:]
y_test_CNN = y_valid_CNN[3000:]
X_train_CNN = X_train_CNN[:8000]
y_train_CNN = y_train_CNN[:8000]
X_valid_CNN = X_valid_CNN[:2000]
y_valid_CNN = y_valid_CNN[:2000]
#y_valid_CNN_normal = y_valid[:2000]

# Acesta este layer-ul care l-am trimis pe kaggle ca raspuns final 
# Declarare model CNN, cu toate layerele
# Model bazat pe LeNet-5
# am incercat sa mai adaug layere dar nu ma ajuta foarte mult din cauza overfit-ului
# iar adaugarea de image augmentation dura prea mult desi normaliza bine.
neural_model = keras.Sequential ([
    layers.Input(shape=(32,32,1)), # avem ca input imaginile sub forma unei matrici
    layers.Conv2D(20, kernel_size=5, padding="same", activation="relu"), # un layer de convolutie cu 20 de filtere
    layers.MaxPooling2D(pool_size=2, strides=2), # un max polling care injumatateste rezolutia filterelor de mai sus
    layers.Dropout(0.2),    #dropout pentru normalizare
    layers.Conv2D(60, kernel_size=5, padding="same", activation="relu"), # Inca un layer convolutional
    layers.Dropout(0.3),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(), # aici transformam filterele de pana acum in vector
    layers.Dense(400, activation='relu'), # Folosirea unui layer dens care merge doar pe date de tip vector, deci evident ca trebuie pus dupa flatten
    #layers.Dense(84, activation='relu'),
    layers.Dense(9, activation="softmax") # softmax returneaza indicele max pe cele 9 "clase" 
    # care sunt de fapt perceptroni, deci returneaza clasa corespunzatoare/ cea mai probabila lui
])

# Early stopping ca sa se opreasca singur cand nu mai progreseaza in antrenare (10 epochs patience)
# Also, avem si restore best weights, deci se intoarece la varianta cea mai buna a acuratetei
early_stopping = callbacks.EarlyStopping(
    min_delta=0.01,
    patience=10,
    restore_best_weights=True
)

# diferiti optimizers pe care i-am mai schimbat din cand in cand, dar in principal am compilat cu Adam
adam = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    amsgrad=False,
    name="Adam"
)
# diferiti optimizeri matematici, mostly used adam
RMSprop = keras.optimizers.RMSprop(
    learning_rate=0.0001,
    rho=0.9,
    momentum=0.0,
    decay=0.0,
    epsilon=1e-07,
    centered=True,
    name="RMSprop"
)

adagrad = keras.optimizers.Adagrad(
    learning_rate=0.04,
    initial_accumulator_value=0.8,
    epsilon=1e-07,
    name="Adagrad"
)
SGD = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.1,
    nesterov=False,
    name="SGD"
)

# Compilarea CNN
neural_model.compile(
    optimizer = adam,
    loss = "categorical_crossentropy", # loss function pentru datele de clasificare
    metrics=['accuracy'] # metrica care ne arata acuratetea la fiecare epoca, pentru train si validare
)
# Obtinerea dictionarului cu date despre antrenare
result = neural_model.fit(
    X_train_CNN,
    y_train_CNN,
    batch_size = 128, # Un batch size de 128 de elemente, am considerat ca 128 e o valoarea optima si pentru testare ~10k values dar si pt 30k(tot)
    validation_data=(X_valid_CNN, y_valid_CNN),  # Validation data pe care se bazeaza output-ul de acuratete la fiecare epoch
    epochs = 100, # folosim 100 de epoci deoarece avem early stopping.
    #shuffle=True,
    #verbose=False,
    use_multiprocessing=True,
    callbacks=[early_stopping] #  Adaugare early stopping
)

# Am incercat aici sa folosesc si un image augmentaiton deoarece cu modelul de mai sus faceam overfit extrem de mult ~2-3 minute pe layer
# Si nu am reusit sa il normalizez nici cu batchnormalisation, iar dropout nu era indeajuns
# Model folosit cu image augmentation-ul de mai jos run time ~= 1 hour, si e underfitted... :(
# Deci aveam underfit cu cateva milioane de parametri, deci a trebuie sa abandonez ideea
# neural_model = keras.Sequential ([
#     layers.Input(shape=(32,32,1)),
#     layers.Conv2D(32, kernel_size=3, activation="relu"),
#     layers.Conv2D(32, kernel_size=3, activation="relu"),
#     layers.Conv2D(32, kernel_size=5, padding="same", activation="relu"),
#     layers.MaxPooling2D(pool_size=2, strides=2),
#     layers.Dropout(0.2),
#     layers.Conv2D(64, kernel_size=3, activation="relu"),
#     layers.Conv2D(64, kernel_size=3, activation="relu"),
#     layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),
#     layers.Dropout(0.3),
#     layers.MaxPooling2D(pool_size=2, strides=2),
#     layers.Flatten(),
#     layers.Dense(1000, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(500, activation='relu'),
#     layers.Dropout(0.1),
#     layers.Dense(9, activation="softmax")
# ])

##training on generator (image augmentation)
#datagen = ImageDataGenerator( # aici am declarat imageDataGenerator care merge specific cu keras.sequential
#    rotation_range=10,  # o rotatie doar de 10, orice valoarea mai mare imi strica totul (nu trecea de 0.15 accuracy)
#    width_shift_range=0.1, # mutare pixeli width-wise de 10%
#    height_shift_range=0.1, #mutare pixeli height-wise de 10%
#    #shear_range=0.2, 
#    zoom_range=0.1, # zooming in with 10%
#    horizontal_flip=False,
#    vertical_flip=False,
#    fill_mode='nearest'
#)
#X_generator = X_train
#X_generator = preprocessor_CNN.transform(X_generator)
#X_generator = X_generator.reshape(-1, 32, 32, 1)
#y_generator = y_train
#y_generator = keras.utils.to_categorical(y_generator, 9)
#datagen.fit(X_generator)
#generator = datagen.flow(X_generator, y_generator, batch_size=128)
#result = neural_model.fit(
#    generator,
#    steps_per_epoch = X_generator.shape[0] // 128,
#    validation_data=(X_valid_CNN, y_valid_CNN),
#    epochs = 100,
#    #shuffle=True,
#    #verbose=False,
#    #use_multiprocessing=True,
#    callbacks=[early_stopping]
#)


# Afisare acuratete
print ("Accuracy: " + str(np.array(result.history['accuracy']).max()))
print ("Val accuracy:" + str(np.array(result.history['val_accuracy']).max()))

# print (X_valid_CNN.shape)
# print (X_valid.shape)
# this X_test e definit mai sus si nu are treaba cu testing-ul de pe kaggle
X_test_CNN = preprocessor_CNN.transform(X_test).reshape((X_test.shape[0], 32, 32, 1))


trainer = neural_model.predict(X_test_CNN)


#obtinere history cu toate datele
history_pd = pd.DataFrame(result.history)
#print (np.argmax(trainer , axis = 1))
#print (np.argmax(y_test_CNN, axis = 1))

# Plotari salvate
plot_1 = history_pd.loc[0:, ['loss', 'val_loss']].plot()
plot_1.figure.savefig('plot_cm_CNN_7_1')
plot_2 = history_pd.loc[0:, ['accuracy', 'val_accuracy']].plot()
plot_2.figure.savefig('plot_cm_CNN_7_2')

# Sender, snipped de code cu care cream fisierul de trimis
# X_test_CNN = preprocessor_CNN.transform(X_test).reshape((X_test.shape[0], 32, 32, 1))
# trainer = neural_model.predict(X_test_CNN)
# 
# trainer = np.argmax(trainer, axis = 1)
# print (trainer)
# #print (np.argmax(trainer , axis = 1))
# #print (np.argmax(y_test_CNN, axis = 1))
# 
# output = pd.DataFrame({'id': X_data_test[0],
#                        'label': trainer})
# print(output.head())
# output.to_csv('submission_CNN_7.cvs', index = False)

# A markdown wrote in the jupyter file, different models and their scores:
#neural_model = keras.Sequential ([
#    layers.Input(shape=(32,32,1)),
#    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#    layers.MaxPooling2D(pool_size=(2, 2)),
#    layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
#    layers.MaxPooling2D(pool_size=(2, 2)),
#    layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
#    layers.MaxPooling2D(pool_size=(2, 2)),
#    layers.Flatten(),
#    layers.Dropout(0.5),
#    layers.Dense(9, activation="softmax")
#])
#  * 0.78-0.79
#neural_model = keras.Sequential ([
#    layers.Input(shape=(32,32,1)),
#    layers.Conv2D(20, kernel_size=5, padding="same", activation="relu"),
#    layers.MaxPooling2D(pool_size=2, strides=2),
#    layers.Dropout(0.2),
#    layers.Conv2D(60, kernel_size=5, padding="same", activation="relu"),
#    layers.Dropout(0.3),
#    layers.MaxPooling2D(pool_size=2, strides=2),
#    layers.Flatten(),
#    layers.Dense(400, activation='relu'),
#    #layers.Dense(84, activation='relu'),
#    layers.Dense(9, activation="softmax")
#])
#    * 0.89%
#    * It's a LeNet-5 arhitecture based. If I can normalize better I might get ~92%
#    * On small data:
#        * Adagrad tops at 0.82, seems to overtrain up to 0.96
#        * adam at 0.82,  seems to overtrain till 0.95
#        * RMSprop at 0.81 and overtrains till 0.92 -- looks like all algs need more normalization

#Trainer with no image augmentation ^
#Epoch 21/100
#235/235 [==============================] - 36s 151ms/step - loss: 0.0715 - accuracy: 0.9770 - val_loss: 0.3397 - val_accuracy: 0.8872
#Accuracy: 0.8928861107145037
#Val accuracy:0.8573714296023051
#```
#With image augmentation it gets the same vals on small dataset, trying big: took around 1-2 hours, gave up the idea of data augmentation

#* ```
#neural_model = keras.Sequential ([
#    layers.Input(shape=(32,32,1)),
#    layers.Conv2D(30, kernel_size=5, padding="same", activation="relu"),
#    layers.MaxPooling2D(pool_size=2, strides=2),
#    layers.Dropout(0.2),
#    layers.Conv2D(60, kernel_size=5, padding="same", activation="relu"),
#    layers.Dropout(0.3),
#    layers.MaxPooling2D(pool_size=2, strides=2),
#    layers.Flatten(),
#    layers.Dense(200, activation='relu'),
#    layers.Dropout(0.3),
#    layers.Dense(84, activation='relu'),
#    layers.Dense(9, activation="softmax")
#])
#Epoch 34/100
#235/235 [==============================] - 64s 274ms/step - loss: 0.1234 - accuracy: 0.9556 - val_loss: 0.3504 - val_accuracy: 0.8872
#Accuracy: 0.8851146137013155
#Val accuracy:0.8660941194085514
#```

#sender 5
#```
#neural_model = keras.Sequential ([
#    layers.Input(shape=(32,32,1)),
#    layers.Conv2D(25, kernel_size=5, padding="same", activation="relu"),
#    layers.MaxPooling2D(pool_size=2, strides=2),
#    layers.Dropout(0.2),
#    layers.Conv2D(85, kernel_size=5, padding="same", activation="relu"),
#    layers.Dropout(0.3),
#    layers.MaxPooling2D(pool_size=2, strides=2),
#    layers.Flatten(),
#    layers.Dense(220, activation='relu'), # + 20 -- revert back, + is usually better
#    layers.Dropout(0.3),
#    layers.Dense(100, activation='relu'), # - 10 -- - is much better
#    layers.Dense(9, activation="softmax")
#])

# Some random saved tries that didn't to that good:
# this method didn't seem that good
# started from leNet-5
# neural_model = keras.Sequential ([
#     layers.Input(shape=(32,32,1)),
#     layers.Conv2D(20, kernel_size=5, padding="same", activation="relu"),
#     layers.MaxPooling2D(pool_size=2, strides=2),
#     # layers.Dropout(0.2),
#     layers.BatchNormalization(),
#
#     layers.Conv2D(20, kernel_size=5, padding="same"),
#     layers.Dropout(0.3),
#     layers.Activation('relu'),
#
#     layers.Conv2D(60, kernel_size=5, padding="same"),
#     layers.Dropout(0.3),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(pool_size=2, strides=2),
#     layers.Flatten(),
#     layers.Dense(400, activation='relu'),
#     # layers.Dropout(0.2),
#     # layers.Dense(84, activation='relu'),
#     layers.Dense(9, activation="softmax")
# ])
# based on VGG16
# neural_model = keras.Sequential ([
#     layers.Input(shape=(32,32,1)),
#     layers.Conv2D(16, kernel_size=5, padding="same", activation="relu"),
#     layers.Conv2D(16, kernel_size=5, padding="same", activation="relu"),
#     layers.MaxPooling2D(pool_size=2, strides=2),
#     layers.Dropout(0.3),
#     layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
#     layers.Dropout(0.3),
#     layers.MaxPooling2D(pool_size=2, strides=2),
#     layers.Flatten(),
#     layers.Dense(256, activation='relu'),
#     #layers.Dense(84, activation='relu'),
#     layers.Dense(9, activation="softmax")
# ])
# neural_model = keras.Sequential ([
#     layers.Input(shape=(32,32,1)),
#     layers.Conv2D(36, kernel_size=5, padding="valid", activation="relu"),
#     layers.Conv2D(72, kernel_size=3,padding='same', strides=2, activation="relu"),
#     layers.Dropout(0.2),
#     layers.AveragePooling2D(pool_size=2, strides=2),
#     layers.Conv2D(144, kernel_size=3, padding="same", activation="relu"),
#     layers.AveragePooling2D(pool_size=2, strides=2),
#     layers.Dropout(0.3),
#     layers.Conv2D(72, kernel_size=3, padding="same", activation="relu"),
#     layers.Dropout(0.2),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(9, activation="softmax")
# ])

