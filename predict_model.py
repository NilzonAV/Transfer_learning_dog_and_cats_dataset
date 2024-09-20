# predict_model.py
import numpy as np
from tensorflow.keras.models import load_model
from src.utils.preprocessing import load_test_images

# Load the trained model
model = load_model('model_v1.h5')
print("Model loaded successfully.")

# Load and preprocess test images
test_images, test_filenames = load_test_images('src/dogs-vs-cats/test1')

# Make predictions
predictions = model.predict(test_images)
predicted_classes = np.where(predictions > 0.5, 1, 0)  # Binary classification

# Display predictions
for filename, prediction in zip(test_filenames, predicted_classes):
    label = 'dog' if prediction == 1 else 'cat'
    print(f"{filename}: {label}")
