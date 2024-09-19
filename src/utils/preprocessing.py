from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import pandas as pd
import os

def create_train_generator(train_path='dogs-vs-cats/train', image_size=(224, 224), batch_size=20):
    """
    Create and return an image data generator for the training dataset with augmentation.
    
    Parameters:
    - train_path: Path to the training dataset directory.
    - image_size: Target size of images (width, height).
    - batch_size: Size of the batches of data.
    """
    # Create a DataFrame
    data = []
    for filename in os.listdir(train_path):
        if filename.startswith('cat'):
            label = 'cat'
        elif filename.startswith('dog'):
            label = 'dog'
        else:
            continue
        data.append({'filename': filename, 'class': label})

    df = pd.DataFrame(data)

    # Create ImageDataGenerator with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,         # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,     # Randomly shift images horizontally by up to 20% of the width
        height_shift_range=0.2,    # Randomly shift images vertically by up to 20% of the height
        shear_range=0.2,           # Randomly apply shearing transformations
        zoom_range=0.2,            # Randomly zoom in by up to 20%
        horizontal_flip=True,      # Randomly flip images horizontally
        fill_mode='nearest'        # Fill in new pixels created during augmentations
    )

    # Create generator using flow_from_dataframe
    train_generator = datagen.flow_from_dataframe(
        df,
        directory=train_path,
        x_col='filename',
        y_col='class',
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size
    )

    return train_generator

def load_test_images(directory, target_size=(224, 224)):
    """
    Load and preprocess test images from a specified directory.
    
    Parameters:
    - directory: Path to the directory containing test images.
    - target_size: Target size of images (width, height).
    """ 
    images = []
    filenames = []
    for fname in sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0])):
        if fname.endswith('.jpg'):
            img_path = os.path.join(directory, fname)
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0) / 255.0
            images.append(img)
            filenames.append(fname)
    return np.vstack(images), filenames
