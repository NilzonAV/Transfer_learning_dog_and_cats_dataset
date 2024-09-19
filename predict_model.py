# train_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.preprocessing import create_train_generator
from tensorflow.keras.regularizers import l2

def get_transfer_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    base_model.trainable = False
    
    # Add L2 regularization to the dense layers
    model = Sequential([
        base_model,
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # Apply L2 regularization
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))  # Apply L2 regularization
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Load training data
train_generator = create_train_generator(train_path='src/dogs-vs-cats/train', batch_size=32)

# Initialize and get the transfer learning model
model = get_transfer_model(input_shape=(224, 224, 3))

# Train the model
history = model.fit(train_generator, epochs=10, 
                    callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])

# Save the trained model
model.save('model_v1.h5')
print("Model training completed and saved as 'model_v1.h5'")
