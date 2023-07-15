# create a benchmark model
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


def get_benchmark_model(input_shape):
    """
    Build and compile a CNN model using the functional API. 
    """
    input_data = Input(shape=input_shape, name="input_img")
    h = Conv2D(32, kernel_size=3, activation="relu", padding="SAME")(input_data)
    h = Conv2D(32, kernel_size=3, activation="relu", padding="SAME")(h)
    h = MaxPooling2D(pool_size=(2,2))(h)
    h = Conv2D(64, kernel_size=3, activation="relu", padding="SAME")(h)
    h = Conv2D(64, kernel_size=3, activation="relu", padding="SAME")(h)
    h = MaxPooling2D(2)(h)
    h = Conv2D(128, kernel_size=3, activation="relu", padding="SAME")(h)
    h = Conv2D(128, kernel_size=3, activation="relu", padding="SAME")(h)
    h = MaxPooling2D(2)(h)
    h = Flatten()(h)
    h = Dense(128, activation="relu")(h)
    output = Dense(1, activation="sigmoid")(h)
    
    model = Model(input_data, output, name="image_classification")
    
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"])
    return model
    
benchmark_model = get_benchmark_model(image_train[0].shape)
benchmark_model.summary()
    
# Train the benchmark model
earlystopping = tf.keras.callbacks.EarlyStopping(patience=2)
history_benchmark = benchmark_model.fit(images_train, labels_train, epoch=10, batch_size=32,
                                            validation_data=(images_valid, labels_valid),
                                            callbacks=[earlystopping])
    
# Evaluate the benchmark model on the test set

benchmark_test_loss, benchmark_test_acc = benchmark_model.evaluate(images_test, labels_test, verbose=0)
print("Test loss: {}".format(benchmark_test_loss))
print("Test accuracy: {}".format(benchmark_test_acc))


# load pretrained images classifier
# the pre-trained model is MobileNet v2 from the keras application datasets.
   
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=None,
        alpha=1.0,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        **kwargs
    )
    
base_model.summary()

# Using the pre-trained model as a feature extractor
# Remove the final layer of the network and replace with the new, untrained classifier for our task.

def remove_head(pretrained_model):
    """This function create and retur a new model

    Args:
        pretrained_model (tensor): _description_
    """
    return Model(pretrained_model.input, pretrained_model.layers[-2].output)

feature_extractor = remove_head(base_model)
feature_extractor.summary()

# Construct the final classifier layers
def add_new_classifier_head(feature_extractor_model):
    """takes the feature extractor model and create and return a new model.

    Args:
        feature_extractor_model (model): _description_
    """
    model = Sequential([
        feature_extractor_model,
        Dense(32, Activation="relu"),
        Dropout(0.5),
        Dense(1, Activation="sigmoit")
    ])
    return model

new_model = add_new_classifier_head(feature_extractor)
new_model.summary()

# Freeze the weights of the pretrained model

def freeze_pretrained_weights(model):
    """freeze the weights of the pretrained base model.
    """
    model.layers[0].trainable = False
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(1e-3),
                  metrics=['accuracy'])
    return model

frozen_new_model = freeze_pretrained_weights(new_model)
frozen_new_model.summary()

# Train the model
earlystopping = tf.keras.callbacks.EarlyStopping(patience=2)
history_frozen_new_model = frozen_new_model(images_train, labels_train, epochs=10, batch_size=32,
                                            validation_data=(images_valid, labels_valid),
                                            callbacks=[earlystopping])

# Compare both models
benchmark_train_loss = history_benchmark.history['loss'][-1]
benchmark_valid_loss = history_benchmark.history['val_loss'][-1]
