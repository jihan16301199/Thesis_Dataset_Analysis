"""
HAM10000_Dataset_RetrievedFrom https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
@author: Jihan Alam
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

dataFrame = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')
#dataFrame.head()

le = LabelEncoder()
le.fit(dataFrame['dx'])
LabelEncoder()
print(list(le.classes_))

dataFrame['label'] = le.transform(dataFrame["dx"]) 
#print(dataFrame.sample(20))

fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(221)
dataFrame['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Type');

plt.tight_layout()
plt.show()

print(dataFrame['label'].value_counts())

#Datapreprocessing
#Creating a balancedDataFrame
 
ndf_0 = dataFrame[dataFrame['label'] == 0]
ndf_1 = dataFrame[dataFrame['label'] == 1]
ndf_2 = dataFrame[dataFrame['label'] == 2]
ndf_3 = dataFrame[dataFrame['label'] == 3]
ndf_4 = dataFrame[dataFrame['label'] == 4]
ndf_5 = dataFrame[dataFrame['label'] == 5]
ndf_6 = dataFrame[dataFrame['label'] == 6]

#ndf_0.head()

sample_size=1000
bdf_0 = resample(ndf_0, replace=True, n_samples=sample_size, random_state=42)
bdf_1 = resample(ndf_1, replace=True, n_samples=sample_size, random_state=42)
bdf_2 = resample(ndf_2, replace=True, n_samples=sample_size, random_state=42)
bdf_3 = resample(ndf_3, replace=True, n_samples=sample_size, random_state=42)
bdf_4 = resample(ndf_4, replace=True, n_samples=sample_size, random_state=42)
bdf_5 = resample(ndf_5, replace=True, n_samples=sample_size, random_state=42)
bdf_6 = resample(ndf_6, replace=True, n_samples=sample_size, random_state=42)

balancedDataFrame = pd.concat([bdf_0,bdf_1,bdf_2,bdf_3,bdf_4,bdf_5,bdf_6])
#print(balancedDataFrame.sample(20))

fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(221)
dataFrame['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Original Data Frame');

ax2 = fig.add_subplot(222)
balancedDataFrame['dx'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count')
ax2.set_title('Balanced Data Frame');

plt.tight_layout()
plt.show()

print(balancedDataFrame['label'].value_counts())

#Adding image path and resized image to the balancedDataFrame  
image_size=64
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('data/HAM10000/', '*', '*.jpg'))}
balancedDataFrame['path'] = dataFrame['image_id'].map(image_path.get)
balancedDataFrame['image'] = balancedDataFrame['path'].map(lambda x: np.asarray(Image.open(x).resize((image_size,image_size))))


X = np.asarray(balancedDataFrame['image'].tolist())
X = X/255.  
Y=balancedDataFrame['label']
Y_cat = to_categorical(Y, num_classes=7)

#Spliting data for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, train_size=0.75, test_size=0.20, random_state=42)

num_classes = 7

model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),activation='relu')) #128
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),activation='relu')) #64
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

batch_size = 16 
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Prediction on test data
y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors
y_true = np.argmax(y_test, axis = 1) 

#Print confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')

