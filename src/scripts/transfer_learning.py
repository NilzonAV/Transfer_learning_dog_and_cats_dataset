import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.ßeras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.preprocessing import create_train_generator, load_test_images

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

# Load training data
train_generator = create_train_generator(train_path='train', batch_size=32)

# Initialize and get the transfer learning model
model = get_transfer_model(input_shape=(224, 224, 3))

# Train the model
history = model.fit(train_generator, epochs=5, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Load and predict test images
test_images, test_filenames = load_test_images('test1')
predictions = model.predict(test_images)
