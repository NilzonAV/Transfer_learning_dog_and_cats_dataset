from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os

def create_train_generator(train_path='train', image_size=(150, 150), batch_size=20):
    """
    Create and return an image data generator for the training dataset.
    
    Parameters:
    - train_path: Path to the training dataset directory.
    - image_size: Target size of images (width, height).
    - batch_size: Size of the batches of data.
    """
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')
    
    return train_generator

def load_test_images(directory, target_size=(150, 150)):
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
