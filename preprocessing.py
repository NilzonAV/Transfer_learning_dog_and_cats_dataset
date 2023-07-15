# Packages imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from tensorflow.keras.models import load_model

# Load and preprocess the data
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
os.getcwd()
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
    
os.listdir(".")

os.mkdir('cats')
os.mkdir('dogs')

images_train = np.load('.../images_train.npy') / 255.0
images_valid = np.load('.../images_valid.npy') / 255.0
images_test = np.load('.../images_test.npy') / 255.0

labels_train = np.load('.../labels_train.npy')
labels_valid = np.load('.../labels_valid.npy')
labels_test = np.load('.../labels_test.npy')

print("{} training data examples".format(images_trian.shape[0]))
print("{} validation data examples".format(images_valid.shape[0]))
print("{} test data examples".format(images_test.shape))

# Display sample images and labels from the training set
class_name = np.array(["Dog", "Cat"])
plt.figure(figsize=(15, 10))
inx = np.random.choice(images_train.shape[0], 15, replace=False)
for n, i in enumerate(inx):
    ax = plt.subplot(3, 5, n+1)
    plt.imshow(images_train[i])
    plt.title(class_name[labels_train[i]])
    plt.axis("off")