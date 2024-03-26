# Transfer Learning

## Overview
This project applies transfer learning techniques using MobileNetV2 to classify images into two categories: dogs and cats. By leveraging a pre-trained neural network, we significantly reduce the time and resources needed for training while maintaining high accuracy.

## Features
- **Transfer Learning**: Utilizes MobileNetV2, a powerful image recognition model pre-trained on ImageNet.
- **Image Preprocessing**: Implements efficient image preprocessing and augmentation to improve model performance.
- **Modular Code**: The project is structured for readability and easy navigation, with separate modules for preprocessing, model training, and visualization.
- **Interactive Visualizations**: Includes plotting utilities to visualize training progress, model accuracy, and loss over time.

## Installation

For Conda:
To set up a local development environment, follow these steps:
conda create --name your_env_name python=3.8
conda activate your_env_name

# install dependencies
pip install -r requirements.txt

# Download the Dataset
The dataset is not included in the GitHub repository due to its size. You can download the dataset from the following link: (https://www.kaggle.com/c/dogs-vs-cats)

After downloading, unzip the dataset into the data/ directory such that you have the following structure:
your_project_name/
├── data/
│   ├── train/
│   └── test1/


#Usage
python src/scripts/transfer_learning.py

#License
This project is licensed under the MIT License.
