# Image Caption Generator

## Introduction
This repository contains code for building and comparing different models for image captioning. The models include LSTM with CNN and CNN with Transformers.

## Dataset
We used a large dataset of images collected from [this GitHub repository](https://github.com/jbrownlee/Datasets).

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Requests
- NLTK
- tqdm

## Contents

- `preprocess_data.py`: Script for data preprocessing.
- `train.py`: Script for training the models.
- `evaluate.py`: Script for evaluating the models.
- `model_comparison.ipynb`: Jupyter notebook for comparing the models' performance.
- `image_captioning.ipynb`: Jupyter Notebook containing the code.
- `features.pkl`: Pickle file containing extracted image features.
- `captions.txt`: Text file containing captions for images.
- `best_model.h5`: Trained model weights.

## Usage

1. Clone the repository.
2. Download and preprocess the dataset.
3. Train the models.
4. Evaluate the models.

## Model Architecture

The models consist of various architectures, including VGG16-based CNN for image feature extraction and LSTM-based RNN for generating captions. Additionally, another model uses Transformer architecture for image captioning.

## Results

The results include evaluation metric scores such as BLEU-1 and BLEU-2. Sample images with their actual and predicted captions are also provided.

## Acknowledgements

- Dataset: [Jason Brownlee](https://github.com/jbrownlee)
- Inspiration: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/12/step-by-step-guide-to-build-image-caption-generator-using-deep-learning/)
