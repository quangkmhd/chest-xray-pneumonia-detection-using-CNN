# Chest X-ray Pneumonia Detection Using Convolutional Neural Networks (CNN)

This project aims to develop a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. By leveraging deep learning techniques, the model assists in the early diagnosis of pneumonia, facilitating timely medical intervention.
## Introduction

Pneumonia is a severe respiratory infection that affects the lungs and can be life-threatening if not diagnosed and treated promptly. Chest X-rays are commonly used for diagnosis; however, interpreting these images requires expertise and can be time-consuming. This project utilizes Convolutional Neural Networks (CNNs) to automate the detection of pneumonia from chest X-ray images, aiming to support healthcare professionals in making accurate and swift diagnoses.

## Features

- **Data Preprocessing**: Includes image resizing, normalization, and augmentation to enhance model performance.
- **Model Architecture**: Utilizes a CNN tailored for image classification tasks.
- **Model Training**: Implements training procedures with techniques like early stopping and learning rate scheduling.
- **Evaluation Metrics**: Assesses model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- **Prediction**: Provides functionality to predict pneumonia presence in new chest X-ray images.

## Dataset

The model is trained and evaluated using the [Chest X-ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. This dataset comprises 5,863 chest X-ray images categorized into:

- **Normal**: Healthy lung images.
- **Pneumonia**: Lung images showing signs of bacterial or viral pneumonia.

Ensure compliance with the dataset's usage terms and patient privacy regulations when utilizing this data.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/quangkmhd/chest-xray-pneumonia-detection-using-CNN.git
   cd chest-xray-pneumonia-detection-using-CNN
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset**:
   - Download the dataset from Kaggle and extract it into the `data/` directory.

2. **Data Preprocessing**:
   - Run the `data_preprocessing.ipynb` notebook to preprocess the images.

3. **Model Training**:
   - Execute the `train_and_test.ipynb` notebook to train the CNN model.

4. **Model Evaluation**:
   - Assess the trained model's performance using the evaluation metrics provided in the notebook.

5. **Prediction**:
   - Use the `prediction.ipynb` notebook to predict pneumonia in new chest X-ray images.

## Project Structure

```
chest-xray-pneumonia-detection-using-CNN/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── train_and_test.ipynb
│   └── prediction.ipynb
├── models/
│   └── cnn_model.h5
└── README.md
```

- `data/`: Contains the dataset organized into training, validation, and test sets.
- `notebooks/`: Includes Jupyter Notebooks for data preprocessing, model training/testing, and prediction.
- `models/`: Stores the trained CNN model.
- `requirements.txt`: Lists the required Python libraries.
- `README.md`: Project documentation.
