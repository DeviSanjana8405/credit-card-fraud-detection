# Credit Card Fraud Detection

This repository contains a machine learning model to detect fraudulent transactions using a dataset of credit card transactions. The project applies various data preprocessing techniques and uses a classifier to identify fraudulent transactions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Modeling Techniques](#modeling-techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud is a major financial issue faced by banks and financial institutions. This project aims to detect fraudulent transactions using machine learning techniques. By analyzing transaction data, the model attempts to identify patterns that signify fraudulent activities.

## Dataset

The dataset used in this project is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013. The dataset consists of 284,807 transactions, out of which 492 are fraudulent.

- **Features:** 30 features (V1 to V28 are PCA-transformed components, Time, Amount)
- **Target:** Binary classification (0 for non-fraud, 1 for fraud)

## Modeling Techniques

The project applies the following steps:

1. **Data Preprocessing:**
   - Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   - Scaling the Amount feature and normalizing the Time feature.
   
2. **Machine Learning Algorithms:**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - XGBoost Classifier

3. **Evaluation Metrics:**
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
   - Area Under ROC Curve (AUC)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

bash
git clone https://github.com/DeviSanjana8405/credit-card-fraud-detection.git


2. Install the required dependencies:

bash
pip install -r requirements.txt


3. Download the dataset and place it in the data/ directory.

## Usage

To run the model and generate predictions:

1. Preprocess the dataset:

bash
python src/data_preprocessing.py


2. Train the model:

bash
python src/model.py


3. Evaluate the model:

bash
python src/evaluation.py


Alternatively, you can explore the Jupyter notebook:

bash
jupyter notebook fraud_detection.ipynb


## Results

The best-performing model was the XGBoost Classifier with the following metrics:

- **Precision:** 0.99
- **Recall:** 0.88
- **F1-Score:** 0.93
- **AUC:** 0.98

More detailed results are available in the notebooks and evaluation.py script.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License.

---
