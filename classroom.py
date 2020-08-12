import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import random
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


DATADIR = "D:/img"
CATEGORIES = ["present", "no_present"]

training_data = []
IMG_SIZE = 100 #nos quedamos con 70 porque era lo minimo que vimos aceptable para diferenciar

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to put together categories
        class_num = CATEGORIES.index(category) #guarda los indices de las categorias 0 1 2 3 4
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                dataset = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([dataset, class_num]) #una imagen y la categoria
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data: #caracteristicas, etiquetas
    X.append(features)#imagenes entrada
    y.append(label) #valores de salida 0 1 2 3 4
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# dataset = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(dataset, cmap = "gray")
# plt.show()

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0 #maximo de valor para las imagenes


model = Sequential() #creamos un modelo básico Sequential

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) #estaba 64 neuronas   (3,3) algo de una window que vamos a usar
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) #fist type of layer

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64)) #64 nodos


#Output layer
model.add(Dense(1)) # 1 neurona de salida dog & cat si se cambia este se debe cambiar el loss paramether
model.add(Activation('sigmoid')) # funcion de activacion queremos usar tansig antes estaba sigmoid

model.compile(loss='binary_crossentropy',# puede usarse sparse_categorical_crossentropy antes binary_crossentropy
              optimizer='adam',
              metrics=['accuracy'])
epochs=5
#hemos probado con 10 y 15 iteraciones pero se sobreajusta
# 32 es la base de batch_size, hemos visto una mejora significativa con 64
history = model.fit(X, y, batch_size=64, epochs=epochs, validation_split=0.02) #10% 
model.save('class.model')
#X and y are what we want to train

#calculate acurracy and error and print
# val_loss, val_acc = model.evaluate(X, y)
# print(val_loss, val_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()