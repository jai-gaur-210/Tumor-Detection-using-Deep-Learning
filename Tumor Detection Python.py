import keras
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout,BatchNormalization,DepthwiseConv2D
from tensorflow.keras import layers

from sklearn.metrics import accuracy_score
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
import os
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
X_train = []
Y_train = []
image_size = 224
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join("E:\\neural project\\Training",i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join("E:\\neural project\\Testing",i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape
X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)
model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu',  padding='same', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))



model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size=(3, 3), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(2, 2))
model.add(DepthwiseConv2D(kernel_size=(3, 3), activation='relu',padding='same'))
model.add(DepthwiseConv2D(kernel_size=(3, 3), activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(layers.GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6,)

history = model.fit(X_train, y_train, epochs=20, validation_split=0.1,callbacks=[reduce_lr])
evaluation_result = model.evaluate(X_test, y_test)
print(f'Test Loss: {evaluation_result[0]:.4f}')
print(f'Test Accuracy: {evaluation_result[1]:.4f}')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

    
model.save("model.h5")