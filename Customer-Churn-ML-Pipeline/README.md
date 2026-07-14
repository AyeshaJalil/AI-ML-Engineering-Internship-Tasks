
Customer Churn Prediction Pipeline
Project Overview

This project builds an end-to-end machine learning pipeline to predict customer churn using the IBM Telco Customer Churn Dataset.

The workflow was developed in Google Colab using Scikit-learn and includes data preprocessing, model training, hyperparameter tuning, evaluation, and pipeline export.

Objective

The main objective is to create a reusable and production-ready customer churn prediction pipeline.

Dataset

The dataset contains:

7,043 rows
21 columns

It includes customer information such as:

Demographics
Services used
Contract type
Payment method
Monthly charges
Total charges
Churn status

The target column is:

Churn

Target values were converted as:

No  → 0
Yes → 1
Data Preprocessing

The following steps were performed:

Removed the customerID column
Converted TotalCharges to numeric
Handled 11 missing values using median imputation
Scaled numerical features using StandardScaler
Encoded categorical features using OneHotEncoder
Used an 80/20 train-test split
Used stratified splitting to preserve churn distribution
Models Used

The following models were trained:

Logistic Regression
Random Forest Classifier
Hyperparameter Tuning

GridSearchCV was used with:

5-fold cross-validation
ROC-AUC scoring
Parallel processing

The best-performing model and hyperparameters were automatically selected.

Model Evaluation

The final model was evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Classification report
Confusion matrix
Results
Metric	Score
Accuracy	Add your score
Precision	Add your score
Recall	Add your score
F1-score	Add your score
ROC-AUC	Add your score
Best Model
Add the model selected by GridSearchCV
Best Hyperparameters
Add grid_search.best_params_ output
Pipeline Export

The complete pipeline was exported using Joblib:

customer_churn_pipeline.joblib

The saved pipeline includes:

Missing-value handling
Numerical scaling
Categorical encoding
Best trained model
Project Files
Customer_Churn_ML_Pipeline.ipynb
customer_churn_pipeline.joblib
evaluation_results.csv
model_comparison.csv
confusion_matrix.png
requirements.txt
README.md
Technologies Used
Python
Google Colab
Pandas
Scikit-learn
Matplotlib
Joblib
GitHub
Skills Demonstrated
Scikit-learn Pipeline API
Data preprocessing
Logistic Regression
Random Forest
GridSearchCV
Hyperparameter tuning
Model evaluation
Model export and reuse
