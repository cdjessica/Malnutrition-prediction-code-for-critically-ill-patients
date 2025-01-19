# -*- coding: utf-8 -*-
"""
@author: CDJessica
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['simsun']
plt.rcParams['axes.unicode_minus'] = False

'''
Read the data and convert data types
'''
data = pd.read_csv('D://DataSet//Model development data.csv')

# Create a column name mapping dictionary
column_mapping = {
    "X9": "BMI",
    "X8": "Age",
    "X5": "Reduced Energy Intake",
    "X15": "Neutrophil Count",
    "X28": "Magnesium Ions",
    "X34": "Total Bilirubin",
    "X30": "Platelet Count",
    "X24": "PaO2",
    "X32": "Serum Creatinine",
    "X19": "IL-6",
    "X22": "hs-CRP",
    "X10": "Total Protein",
    "X18": "Fasting Blood Glucose Value",
    "X12": "Hemoglobin",
    "X11": "Albumin",
    "X23": "pH Value",
    "X26": "Sodium Ions"
}

# Rename columns
data.rename(columns=column_mapping, inplace=True)

# Output the information of the modified data frame
print(data.head())
print(data.info())

# Convert columns of object type to numeric type
object_columns = data.select_dtypes(include=['object']).columns

## Loop through each column
for col in object_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Display the converted DataFrame
print(data)
data.info()
Variable_Name = data.columns 

'''
Impute missing values using the Random Forest algorithm
'''
# Check for missing values in each variable
print(pd.isna(data).sum())

# Visualize the distribution of missing values in the data
import missingno as msno
msno.matrix(data, figsize=(16, 10), width_ratios=(13, 2), color=(0.25, 0.25, 0.5))
plt.show()

# Remove columns with fewer than 30 non-null elements
data = data.dropna(axis='columns', thresh=30)

# Check the data types
print(data.dtypes)

# List of variables without missing values
estimate = ['X1', 'X2', 'X3', 'X4',
            'Reduced Energy Intake',  'X6', 'X7', 'Age',  'Total Protein', 'Albumin',
            'Hemoglobin', 'X13', 'X14', 'Neutrophil Count',
            'X17', 'Fasting Blood Glucose Value',
            'X25', 'Sodium Ions', 'X27', 'Platelet Count', 'X31', 'Serum Creatinine', 'X33', 'Total Bilirubin',
            'X35', 'X36', 'X37', 'X38', 'X39', 'Y']


# Impute missing values using Random Forest
def set_missing(df, estimate_list, miss_col):
    col_list = estimate_list
    col_list.append(miss_col)
    process_df = df.loc[:, col_list]
    class_le = LabelEncoder()
    for i in col_list[:-1]:
        process_df.loc[:, i] = class_le.fit_transform(process_df.loc[:, i].values)
    known = process_df[process_df[miss_col].notnull()].values
    known[:, -1] = class_le.fit_transform(known[:, -1])
    unknown = process_df[process_df[miss_col].isnull()].values
    # X represents the feature attribute values
    X = known[:, :-1]
    # y represents the outcome label values
    y = known[:, -1]
    # fit the data into a RandomForestRegressor
    rfr = RandomForestRegressor(random_state=2, n_estimators=200, max_depth=4, n_jobs=-1)
    rfr.fit(X, y)
    # Use the obtained model to predict unknown feature values
    predicted = rfr.predict(unknown[:, :-1]).round(0).astype(int)
    predicted = class_le.inverse_transform(predicted)
    #print(predicted)
    # Impute the original missing data using the obtained prediction results
    df.loc[(df[miss_col].isnull()), miss_col] = predicted
    return df

set_missing(data, estimate, 'BMI')
set_missing(data, estimate, 'X16')
set_missing(data, estimate, 'IL-6')
set_missing(data, estimate, 'X20')
set_missing(data, estimate, 'X21')
set_missing(data, estimate, 'hs-CRP')
set_missing(data, estimate, 'pH Value')
set_missing(data, estimate, 'PaO2')
set_missing(data, estimate, 'Magnesium Ions')
set_missing(data, estimate, 'X29')

data.isnull().any() # Check for any missing values after imputation
data.describe()

# Split the dataset into features and target variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

feature_name = X.columns # Feature names

## Split the dataset into training and testing sets in an 8:2 ratio
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19888, random_state=42)
#X_train.to_excel('C://Users//Singularity//Desktop//X_train.xlsx')
#X_test.to_excel('C://Users//Singularity//Desktop//X_test.xlsx')
#y_train.to_excel('C://Users//Singularity//Desktop//y_train.xlsx')
#y_test.to_excel('C://Users//Singularity//Desktop//y_test.xlsx')

# Standardize the continuous variables
scaler = StandardScaler()
X.iloc[:, 7:] = scaler.fit_transform(X.iloc[:, 7:])

# Check the balance of classification labels
from collections import Counter
print(Counter(y))

Downsample using the RandomUnderSampler method (prototype selection data)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = rus.fit_resample(X, y)
print(Counter(y_resampled))

'''
Random Forest RFECV feature selection
'''
from sklearn.feature_selection import RFECV

# Create a Random Forest Classifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)  
  
# Create an RFECV instance
rfecv = RFECV(estimator=RFC, step=1, cv=5, scoring='accuracy')  
  
# Fit the model
rfecv.fit(X, y)  
  
# 打印选择的特征数量  
print("Optimal number of features : %d" % rfecv.n_features_)  
  
# Obtain the feature set at the best step
best_features = rfecv.support_  
  
Retrieve the feature importance at the best step (from the final estimator)
best_feature_importance = rfecv.estimator_.feature_importances_  
best_feature_importance = best_feature_importance[:rfecv.n_features_]  # Keep only the importance of the selected features
  
# Obtain the names of the features at the best step  
feature_names = X_resampled.columns  
best_feature_names = [feature_names[i] for i in range(len(feature_names)) if best_features[i]]  
  
Print the feature names and their importance at the best step
print("Best Features:", best_feature_names)  
print("Best Feature Importance:", best_feature_importance)  
  
Rank features by importance
sorted_indices = np.argsort(best_feature_importance)  
sorted_features = [best_feature_names[i] for i in sorted_indices]  
sorted_importance = best_feature_importance[sorted_indices]  
  
Print the features ranked by importance
print("Feature Importance Ranking:", sorted_features)  
  
# Visualize the performance at each step of RFECV
plt.figure(figsize=(10, 6))  
plt.xlabel("Number of Selected Features")  
plt.ylabel("Cross-validation Score")  
  
# Obtain the cross-validation scores (mean_test_score)
n_features = np.arange(1, len(rfecv.cv_results_['mean_test_score']) + 1)  
scores = rfecv.cv_results_['mean_test_score']  
plt.plot(n_features, scores, marker='o')  
plt.scatter(rfecv.n_features_ + 1, np.max(scores), c='red', marker='o', label='Best Step')
plt.legend()  
plt.title("RFECV - Feature Selection") 
plt.grid(True)  
plt.show()  
  
Visualize the features ranked by importance
plt.figure(figsize=(10, 6))  
plt.barh(sorted_features, sorted_importance, color='skyblue')  
plt.xlabel("Feature Importance")   #, fontproperties='SimHei'
plt.ylabel("Feature Variables")   #, fontproperties='SimHei'
# plt.title("Feature Importance Ranking")  
plt.show()


Obtain the optimal feature subset
Obtain the boolean mask of selected features
feature_names = X.columns
feature_mask = rfecv.support_

# Obtain the names of the selected features
selected_features = [feature for feature, mask in zip(feature_names, feature_mask) if mask]

Print the names of the selected features
print("Selected Features:", selected_features)

# Select important features
X_selected = rfecv.transform(X)
X_selected = pd.DataFrame(X_selected)
X_selected.columns = selected_features

# Split the dataset into training and testing sets in an 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.19888, random_state=42)

# Print the evaluation metrics for the KNN, SVM, Decision Tree, Random Forest, Gaussian Naive Bayes, LR, and XGBoost models
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier()
}

from sklearn.metrics import precision_recall_curve, average_precision_score

for name, clf in classifiers.items():
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict probabilities - training set
    y_train_prob = clf.predict_proba(X_train)
    
    # Print the evaluation metrics - training set
    print(f"\n{name} Evaluation on Training Set:")
    print(classification_report(y_train, clf.predict(X_train)))
    
    # ROC curve and AUC for each class - training set
    for i in range(len(clf.classes_)):
        fpr_train, tpr_train, _ = roc_curve(y_train == clf.classes_[i], y_train_prob[:, i])
        roc_auc_train = auc(fpr_train, tpr_train)
        
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_train:.2f}) for class {clf.classes_[i]} - Training Set')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("1-Specificity", fontproperties='simsun')
        plt.ylabel("Sensitivity", fontproperties='simsun')
        plt.title(f'{name} - ROC Curve - Class {clf.classes_[i]} - Training Set', fontproperties='simsun')
        plt.legend(loc="lower right")
        plt.show()
    
    # PR curve for each class - training set
    for i in range(len(clf.classes_)):
        precision_train, recall_train, _ = precision_recall_curve(y_train == clf.classes_[i], y_train_prob[:, i])
        avg_precision_train = average_precision_score(y_train == clf.classes_[i], y_train_prob[:, i])
        
        plt.figure()
        plt.step(recall_train, precision_train, color='b', alpha=0.2, where='post')
        plt.fill_between(recall_train, precision_train, step='post', alpha=0.2, color='b')
        plt.xlabel("Recall", fontproperties='simsun')
        plt.ylabel("Precision", fontproperties='simsun')
        plt.title(f'{name} - PR Curve - Class {clf.classes_[i]} (Avg Precision = {avg_precision_train:.2f}) - Training Set', fontproperties='simsun')
        plt.show()

for name, clf in classifiers.items():
    # Train the models
    clf.fit(X_train, y_train)
    
    # Predict probabilities
    y_test_prob = clf.predict_proba(X_test)
    
    # 打印评价指标
    print(f"\n{name} Evaluation:")
    print("Training Set:")
    print(classification_report(y_train, clf.predict(X_train)))
    print("Testing Set:")
    print(classification_report(y_test, clf.predict(X_test)))
    
    # ROC curve and AUC for each class
    for i in range(len(clf.classes_)):
        fpr, tpr, _ = roc_curve(y_test == clf.classes_[i], y_test_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for class {clf.classes_[i]} - Test Set')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False positive rate", fontproperties='simsun')
        plt.ylabel("True positive rate", fontproperties='simsun')
        plt.title(f'{name} - ROC Curve - Class {clf.classes_[i]}', fontproperties='simsun')
        plt.legend(loc="lower right")
        plt.show()
    
    # PR curve for each class
    for i in range(len(clf.classes_)):
        precision, recall, _ = precision_recall_curve(y_test == clf.classes_[i], y_test_prob[:, i])
        avg_precision = average_precision_score(y_test == clf.classes_[i], y_test_prob[:, i])
        
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel("Recall", fontproperties='simsun')
        plt.ylabel("Precision", fontproperties='simsun')
        plt.title(f'{name} - P-R Curve - Class {clf.classes_[i]} (Avg Precision = {avg_precision:.2f}) - Test Set', fontproperties='simsun')
        plt.show()
        
'''
Compare models horizontally
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

import random
random.seed(42)

import numpy as np
np.random.seed(42)

# Define the models dictionary
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier()
}

# Define hyperparameters grid
param_grids = {
    'KNN': {'n_neighbors': [3, 5, 7]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [None, 5, 10, 20]},
    'Random Forest': {'n_estimators': [50, 100, 200]},
    'Naive Bayes': {},
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'XGBoost': {'n_estimators': [50,100,200], 'max_depth': [3, 5, 7, 9], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Store the best models and accuracy
best_models = {}

# Store the accuracy, precision, recall, and F1 score
accuracies = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}

# Iterate through the model dictionary to perform grid search
for model_name, model in classifiers.items():
    print(f"Searching for the best hyperparameters for {model_name}...")

    # Perform grid search using GridSearchCV
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Obtain the best model
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model

    # Perform predictions on the testing set
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate macro precision, macro recall, and macro F1 score
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Store in a dictionary
    accuracies['Model'].append(model_name)
    accuracies['Accuracy'].append(accuracy)
    accuracies['Precision'].append(precision)
    accuracies['Recall'].append(recall)
    accuracies['F1'].append(f1)

# Convert the accuracy table to a DataFrame
accuracy_df = pd.DataFrame(accuracies)

# Print the table of accuracy, precision, recall, and F1 score
print(accuracy_df)


# Plot ROC and PR curves on the same figure for comparison
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Define the models
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier()
}

# Parameter grid
param_grids = {
    'KNN': {'n_neighbors': [3, 5, 7]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [None, 5, 10, 20]},
    'Random Forest': {'n_estimators': [50, 100, 200]},
    'Naive Bayes': {},
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'XGBoost': {'n_estimators': [50,100,200], 'max_depth': [3, 5, 7, 9], 'learning_rate': [0.01, 0.1, 0.2]}
}

from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Plot the PR curve
plt.figure(figsize=(6, 6))

# Loop through each model
for model_name, model in classifiers.items():
    
    # Grid search cross-validation for hyperparameter tuning
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Output the best parameters
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    # Train the model using the best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Obtain the probability predictions from the model
    y_score = best_model.predict_proba(X_test)
    
    # Binzarize the labels
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    # Calculate the PR curve
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    pr_auc = auc(recall, precision)

    # Plot the PR curve
    plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')

# Plot the diagonal
plt.plot([0, 1], [1, 0], 'k--', lw=2)
plt.xlabel('Recall', fontproperties='simsun')
plt.ylabel('Precision', fontproperties='simsun')
plt.title('PR Curve', fontproperties='simsun')
plt.legend(loc='lower left')
plt.grid(True)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# Plot the ROC curve
plt.figure(figsize=(6, 6))  

# Loop through each model
for model_name, model in classifiers.items():
    # Perform grid search cross-validation for hyperparameter tuning
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Output the best parameters
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    # Train the model using the best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Obtain the probability predictions from the model
    y_score = best_model.predict_proba(X_test)
    
    # Binzarize the labels
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot the diagonal
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('1-Specificity', fontproperties='simsun')
plt.ylabel('Sensitivity', fontproperties='simsun')
plt.title('ROC Curve', fontproperties='simsun')
plt.legend(loc='lower right')
plt.grid(True)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()

for i in range(n_classes):
    plt.figure(figsize=(12, 6))

    # Plot the calibration curve for the training set
    for name, model in classifiers.items():
        print(f"Training and calibrating {name} for Class {i} (Training Set)...")

        # If the model has a parameter grid, perform grid search
        if param_grids.get(name):
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            best_model = model

        # Calibrate the best model using Platt Scaling with CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv='prefit')
        best_model.fit(X_train, y_train)
        calibrated_model.fit(X_train, y_train)
        
        # Obtain the prediction probabilities for the training set
        y_train_prob = calibrated_model.predict_proba(X_train)[:, i]

        # Calculate the calibration curve
        prob_true, prob_pred = calibration_curve(y_train == i, y_train_prob, n_bins=10)
        ece = expected_calibration_error(y_train == i, y_train_prob)
        brier = bCalculate the calibrationrier_score_loss(y_train == i, y_train_prob)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{name} (Brier: {brier:.4f}, ECE: {ece:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Bin\'s mean of predicted probability')
    plt.ylabel('Bin\'s mean of target variable')
    plt.title(f'Training Set Calibration plot for Class {i}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))

    # Plot the calibration curve for the test set
    for name, model in classifiers.items():
        print(f"Training and calibrating {name} for Class {i} (Test Set)...")

        # If the model has a parameter grid, perform grid search
        if param_grids.get(name):
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model

        # Calibrate the best model using Platt Scaling with CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv='prefit')
        best_model.fit(X_train, y_train)
        calibrated_model.fit(X_train, y_train)

        # Obtain the prediction probabilities for the test set
        y_test_prob = calibrated_model.predict_proba(X_test)[:, i]

        # Calculate the calibration curve
        prob_true, prob_pred = calibration_curve(y_test == i, y_test_prob, n_bins=10)
        ece = expected_calibration_error(y_test == i, y_test_prob)
        brier = brier_score_loss(y_test == i, y_test_prob)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{name} (Brier: {brier:.4f}, ECE: {ece:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Bin\'s mean of predicted probability')
    plt.ylabel('Bin\'s mean of target variable')
    plt.title(f'Test Set Calibration Plot for Class {i}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()