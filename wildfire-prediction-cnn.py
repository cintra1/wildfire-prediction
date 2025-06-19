import kagglehub
abdelghaniaaba_wildfire_prediction_dataset_path = kagglehub.dataset_download('abdelghaniaaba/wildfire-prediction-dataset')

print('Data source import complete.')
import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPooling2D,Input,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2

dir = '/kaggle/input/wildfire-prediction-dataset/train'
x = []
y = []
for direct in os.listdir(dir):
    print("Loading dataset training {}".format(direct))
    for filename in os.listdir(os.path.join(dir,direct)):
        img_path = os.path.join(dir,direct,filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32,32))
        img = np.array(img)
        img = img/255
        x.append(img)
        y.append(direct)


dir_val = '/kaggle/input/wildfire-prediction-dataset/valid'
x_val=[]
y_val=[]
for direct in os.listdir(dir_val):
    print("Loading dataset validation {}".format(direct))
    for filename in os.listdir(os.path.join(dir_val,direct)):
        img_path = os.path.join(dir_val,direct,filename)
        image = cv2.imread(img_path)
        image = cv2.resize(image,(32,32))
        image = np.array(image)
        image = image/255
        x_val.append(image)
        y_val.append(direct)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
le = LabelEncoder()
int_label = le.fit_transform(y)
one_hot = to_categorical(int_label)

print(int_label)
print(one_hot)

le = LabelEncoder()
int_label = le.fit_transform(y_val)
one_hot_val = to_categorical(int_label)

x = np.array(x)
x_val = np.array(x_val)

from sklearn.utils import shuffle
x,one_hot = shuffle(x,one_hot)
x_val,one_hot_val = shuffle(x_val,one_hot_val)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,one_hot,test_size=0.2)

Y_train = np.array([np.array(i) for i in Y_train])
Y_test = np.array([np.array(i) for i in Y_test])
one_hot_val = np.array([np.array(i) for i in one_hot_val])

from keras import layers
from keras import models
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history=model.fit(X_train,Y_train,validation_data=(x_val,one_hot_val),batch_size=32,epochs=10)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.show()

import matplotlib.image as mping
test_img_path = "/kaggle/input/wildfire-prediction-dataset/test/wildfire/-59.03238,51.85132.jpg"
img = mping.imread(test_img_path)
imgplot = plt.imshow(img)
plt.xlabel("Wildfire")
plt.show()

test_arr = []
test_image = cv2.imread(test_img_path)
test_image = cv2.resize(test_image,(32,32))
test_image = np.array(test_image)
test_image = test_image/255
test_image = test_image.reshape(1,32,32,3)
test_arr.append(test_image)

prediction = model.predict(test_arr) 
predicted_class = np.argmax(prediction)
print("Classe prevista:", le.classes_[predicted_class])