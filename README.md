# Transfer Learning with MobileNetV2: Dog vs. Cat Classification

# Overview

This project implements a Transfer Learning approach using the pre-trained MobileNetV2 model to classify images into two categories: dogs and cats. By leveraging a pre-trained model, the training time and computational resources are significantly reduced while achieving high classification accuracy.

# Features

Transfer Learning: Utilizes the MobileNetV2 model pre-trained on ImageNet to adapt quickly to the dog vs. cat classification task.
Image Preprocessing & Augmentation: Efficient preprocessing and augmentation techniques such as resizing, scaling, and flipping are used to improve model generalization and performance.
Modular Architecture: The project is designed with a clean, modular structure that separates concerns into data preprocessing, model training, and result visualization.
Visualization Tools: Includes utilities to generate plots of training and validation accuracy and loss over time to monitor model performance.

# Project Structure

your_project_name/

├── README.md

├── predict_model.py
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── models
│   │   └── __init__.py
│   ├── scripts
│   │   └── __init__.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-312.pyc
│       │   ├── plotting.cpython-312.pyc
│       │   └── preprocessing.cpython-312.pyc
│       ├── plotting.py
│       └── preprocessing.py
└── train.py

Before running the project, ensure you have the following dependencies installed.

Setting up Conda Environment
# Create a virtual environment with Python 3.8
conda create --name your_env_name python=3.8

# Activate the environment
conda activate your_env_name

# Install dependencies
pip install -r requirements.txt
Installing Dependencies
If you're not using Conda, you can install the required Python libraries using pip:
pip install -r requirements.txt
Dataset

The dataset is not included in this repository due to its size. You can download the dataset from Kaggle: Dogs vs. Cats Dataset.

After downloading the dataset, unzip the files and place them into the following directory structure:

your_project_name/

├── data/
│   ├── train/        # Training images (cat and dog images)
│   └── test1/        # Testing images
Usage

Once the dataset is set up, you can train the model by running the following command:
python src/scripts/transfer_learning.py
This script will preprocess the data, train the model using transfer learning, and save the trained model in the models/ directory.

Predicting on New Images
To make predictions on new images using the trained model, run the predict_model.py script:
python predict_model.py
Visualizations

The project includes utilities to plot learning curves (accuracy and loss) over time, helping you monitor the model's performance during training.

Example Learning Curves:
The plotting.py script will generate and display plots for:

Training vs. Validation Accuracy
Training vs. Validation Loss
You can view these curves by checking the saved plots or generating them during training.

License

This project is licensed under the MIT License. See the LICENSE file for details.
