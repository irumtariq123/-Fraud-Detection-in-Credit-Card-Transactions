Credit Card Fraud Detection

Project Overview:

This project aims to develop an automated fraud detection system that predicts whether a credit card transaction is fraudulent. By analyzing transaction data, the model assists financial institutions in detecting potential fraud and minimizing financial losses.

Table of Contents:
Project Overview
Dataset
Data Preprocessing
Modeling
Evaluation
Results
Improvements
Future Work

Dataset:

The dataset contains anonymized transaction data with features that describe each transaction, such as transaction amount, time, and various other factors. It includes a highly imbalanced binary target class, where 1 represents fraudulent transactions and 0 represents legitimate transactions.

Data Preprocessing:

Handling Imbalanced Data: The dataset is highly imbalanced. We used SMOTE (Synthetic Minority Over-sampling Technique) and ENN (Edited Nearest Neighbors) to balance the dataset.
Feature Scaling: We applied standard scaling to ensure features are normalized for models like Logistic Regression.

Modeling:

We experimented with several machine learning models:

Logistic Regression
Random Forest
XGBoost

Hyperparameter Tuning:

Grid Search and Random Search were used to fine-tune the models for better performance.
Evaluation

We evaluated the models using:

Confusion Matrix
ROC-AUC Score
Precision, Recall, and F1-score
Due to the imbalanced nature of the dataset, these metrics help in understanding the true model performance beyond accuracy.

Results:

Initial Models: Logistic Regression and Random Forest struggled with the imbalanced data, showing low fraud detection.
Improved Models: After applying data balancing and advanced models like XGBoost, fraud detection improved significantly, as indicated by higher ROC-AUC and better Precision-Recall scores.

Improvements:

Applied SMOTE + ENN for effective data balancing.
Introduced XGBoost, which proved more effective for fraud detection.
Used cross-validation and hyperparameter tuning for better model performance.

Future Work:
Further hyperparameter tuning and feature engineering.
Deploy the model as a real-time fraud detection system.
Incorporate more advanced techniques like deep learning models for large-scale datasets.
