# Plot the learning curves
import matplotlib.pyplot as plt
import numpy as np
import random

def display_sample_images(images_train, labels_train, class_names=np.array(["Dog", "Cat"]), sample_size=15):
    """
    Display a sample of images from the training set.

    Parameters:
    - images_train: numpy array of training images.
    - labels_train: numpy array of image labels.
    - class_names: list of class names corresponding to labels.
    - sample_size: number of images to display.
    """
    plt.figure(figsize=(15, 10))
    idx = np.random.choice(images_train.shape[0], sample_size, replace=False)
    for n, i in enumerate(idx):
        ax = plt.subplot(3, 5, n+1)
        plt.imshow(images_train[i])
        plt.title(class_names[labels_train[i]])
        plt.axis("off")
    plt.show()

def plot_training_history(history, history_title='model'):
    """
    Plot the accuracy and loss graphs for the training and validation sets.

    Parameters:
    - history: Training history object returned by model.fit().
    - history_title: Title to distinguish different models' histories if needed.
    """
    accuracy = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_accuracy = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    
    plt.figure(figsize=(15,5))

    # Plot accuracy
    plt.subplot(121)
    plt.plot(history.history[accuracy])
    plt.plot(history.history[val_accuracy])
    plt.title(f'Accuracy vs. Epochs ({history_title})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # Plot loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Loss vs. Epochs ({history_title})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.show()

