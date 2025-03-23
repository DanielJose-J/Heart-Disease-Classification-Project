Heart Disease Prediction using Machine Learning

This project demonstrates how to use various machine learning algorithms to predict the presence of heart disease based on clinical and demographic features. The workflow includes data preprocessing, exploratory data analysis, model building, evaluation, and interpretation.

🚀 Project Overview

Goal: Predict whether a patient has heart disease using their medical attributes.

Dataset: Heart Disease UCI dataset (Cleveland data)

📂 Contents

Data Loading and Cleaning

Exploratory Data Analysis (EDA)

Feature Understanding and Visualization

Model Training (Logistic Regression, K-Nearest Neighbors, Random Forest)

Hyperparameter Tuning (RandomizedSearchCV, GridSearchCV)

Model Evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

Feature Importance Analysis

📊 Dataset Info

303 samples

14 attributes including age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.

Target variable: presence of heart disease (1 = yes, 0 = no)

🔧 Requirements

Python 3.x

Libraries:

numpy

pandas

matplotlib

seaborn

scikit-learn

🧠 Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC Curve and AUC Score

📌 Key Insights

Logistic Regression and Random Forest outperformed KNN in accuracy and F1 score.

Feature importance was interpreted using model coefficients and visualizations.

📚 Acknowledgments

UCI Machine Learning Repository

Scikit-learn Documentation

