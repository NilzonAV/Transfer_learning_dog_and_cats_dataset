import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.plotting import plot_training_history
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

def get_transfer_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    base_model.trainable = False
    model = Sequential([
        base_model,
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the path to the dataset directory
train_path = 'src/dogs-vs-cats/train'

# Create a DataFrame with the filenames and labels
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

# Create ImageDataGenerator with validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Specify validation split

# Create training generator
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_path,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Set as training data
)

# Create validation generator
validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_path,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Set as validation data
)

# Initialize and get the transfer learning model
model = get_transfer_model(input_shape=(224, 224, 3))

# Train the model using the separate validation generator
history = model.fit(train_generator, epochs=5, validation_data=validation_generator,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Save the trained model
model.save('model_v1.h5')
print("Model training completed and saved as 'model_v1.h5'")

# Plot the training history to visualize learning curves
plot_training_history(history, history_title='Transfer Learning Model')
