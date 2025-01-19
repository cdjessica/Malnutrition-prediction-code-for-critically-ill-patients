# -*- coding: utf-8 -*-
"""
@author: CDJessica
"""
# XGBoost external validation data
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
Read the data and perform data type conversion
'''
data = pd.read_csv('D://DataSet//Model development data.csv')

# Create a column name mapping dictionary
column_mapping = {
    "X9": "BMI",
    "X8": "Age",
    "X5": "Reduced Intake",
    "X15": "Neutrophils",
    "X28": "Magnesium Ion",
    "X34": "Total Bilirubin",
    "X30": "Platelets",
    "X24": "PaO2",
    "X32": "Serum Creatinine",
    "X19": "IL-6",
    "X22": "hs-CRP",
    "X10": "Total Protein",
    "X18": "Fasting Blood Glucose",
    "X12": "Hemoglobin",
    "X11": "Albumin",
    "X23": "pH value",
    "X26": "Sodium Ion"
}

# Rename columns
data.rename(columns=column_mapping, inplace=True)

# Output the information of the modified data frame
print(data.head())
print(data.info())

# Convert columns of object type to numeric type
object_columns = data.select_dtypes(include=['object']).columns

# Loop through each column
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
            'Reduced Intake',  'X6', 'X7', 'Age',  'Total Protein', 'Albumin',
            'Hemoglobin', 'X13', 'X14', 'Neutrophils',
            'X17', 'Fasting Blood Glucose',
            'X25', 'Sodium Ion', 'X27', 'Platelets', 'X31', 'Serum Creatinine', 'X33', 'Total Bilirubin',
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
set_missing(data, estimate, 'pH value')
set_missing(data, estimate, 'PaO2')
set_missing(data, estimate, 'Magnesium Ion')
set_missing(data, estimate, 'X29')

data.isnull().any() # Check for any missing values after imputation
data.describe()

# Split the dataset into features and target variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

feature_name = X.columns # Feature names

# Split the dataset into training and testing sets in an 8:2 ratio
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

# Print the number of selected features
print("Optimal number of features : %d" % rfecv.n_features_)

# Obtain the feature set at the best step
best_features = rfecv.support_

# Obtain the feature importance at the best step
best_feature_importance = rfecv.estimator_.feature_importances_

# Obtain the names of the features at the best step
feature_names = X.columns
best_feature_names = [feature_names[i] for i in range(len(feature_names)) if best_features[i]]

# Print the feature names and their importance at the best step
print("Best Features:", best_feature_names)
print("Best Feature Importance:", best_feature_importance)

# Rank features by importance
sorted_indices = np.argsort(best_feature_importance)
sorted_features = [best_feature_names[i] for i in sorted_indices]
sorted_importance = np.sort(best_feature_importance)

# Obtain the optimal feature subset
# Obtain the boolean mask of selected features
feature_names = X.columns
feature_mask = rfecv.support_

# Obtain the names of the selected features
selected_features = [feature for feature, mask in zip(feature_names, feature_mask) if mask]

# Print the names of the selected features
print("Selected Features:", selected_features)

# Select important features
X_selected = rfecv.transform(X)
X_selected = pd.DataFrame(X_selected)
X_selected.columns = selected_features

# Split the dataset into training and testing sets in an 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.19888, random_state=42)
#X_train.to_excel('C://Users//Singularity//Desktop//X_train.xlsx')
#X_test.to_excel('C://Users//Singularity//Desktop//X_test.xlsx')
#y_train.to_excel('C://Users//Singularity//Desktop//y_train.xlsx')
#y_test.to_excel('C://Users//Singularity//Desktop//y_test.xlsx')

# # Define the XGBoost model (using default parameters)
# xgb_model = XGBClassifier()

# # Train the XGBoost model
# xgb_model.fit(X_train, y_train)

# # Evaluate the model on the test set
# y_pred = xgb_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test Accuracy: {:.4f}".format(accuracy))

# Train the XGBoost model using grid search
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Define the XGBoost model
xgb_model = XGBClassifier()


# Define the hyperparameter grid
param_grids = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [3, 5, 7, 9],       
    'learning_rate': [0.01, 0.1, 0.2], 

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grids, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Train the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Output the best parameters and the best model
print("Best Parameters:", grid_search.best_params_)
xgb_model = grid_search.best_estimator_

# # Evaluate the best model on the testing set
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test Accuracy: {:.4f}".format(accuracy))


# Validate the XGBoost model using an external dataset
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, average_precision_score, classification_report
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Load the dataset
data = pd.read_excel(r'C:\Users\Administrator\External validation dataset.xlsx')

# Extract features and target variable
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Align the column names of the external validation data with those of the training data
X = X.rename(columns={
    'Reduced Energy Intake': 'Reduced Intake',
    'Sodium Ions': 'Sodium Ion',
    'Fasting Blood Glucose Value': 'Fasting Blood Glucose',
    'Magnesium Ions': 'Magnesium Ion',
    'Neutrophil Count': 'Neutrophils',
    'Platelet Count': 'Platelets',
    'pH Value': 'pH value'
})


# Reorder the columns of the external validation data to ensure the column order matches that of the training data
X = X[xgb_model.feature_names_in_]

# Standardize the continuous variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Perform predictions on the external validation dataset
y_pred = xgb_model.predict(X)
y_pred_prob = xgb_model.predict_proba(X)

# Output the classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# ROC curve and AUC for each class
for i in range(len(xgb_model.classes_)):
    fpr, tpr, _ = roc_curve(y == xgb_model.classes_[i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for class {xgb_model.classes_[i]} - External Validation Set')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'ROC Curve - Class {xgb_model.classes_[i]}')
    plt.legend(loc="lower right")
    plt.show()

# PR curve and AUC for each class
for i in range(len(xgb_model.classes_)):
    precision, recall, _ = precision_recall_curve(y == xgb_model.classes_[i], y_pred_prob[:, i])
    avg_precision = average_precision_score(y == xgb_model.classes_[i], y_pred_prob[:, i])
    
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f'P-R Curve - Class {xgb_model.classes_[i]} (Avg Precision = {avg_precision:.2f}) - External Validation Set')
    plt.show()


# Plot the calibration curve for external validation of XGBoost
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_accuracy = np.mean(y_true[bin_mask] == 1)
            bin_confidence = np.mean(y_prob[bin_mask])
            ece += np.abs(bin_accuracy - bin_confidence) * len(y_true[bin_mask]) / len(y_true)
    return ece

# Plot the calibration curve for the external validation dataset
from sklearn.metrics import brier_score_loss
plt.figure(figsize=(10, 7))
for i in range(y_pred_prob.shape[1]):
    prob_true, prob_pred = calibration_curve(y == i, y_pred_prob[:, i], n_bins=10)
    
    # Calculate the Brier scores
    brier = brier_score_loss(y == i, y_pred_prob[:, i])
    
    # Calculate the ECE values
    ece = expected_calibration_error(y == i, y_pred_prob[:, i])

    # Plot the calibration curve for each class
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i} (Brier: {brier:.4f}, ECE: {ece:.4f})')

# Plot the diagonal
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Bin\'s mean of predicted probability')
plt.ylabel('Bin\'s mean of target variable')
plt.title('XGBoost External Validation Set Calibration Plot (Multi-class)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

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
brier = brier_score_loss(y_train == i, y_train_prob)
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

# Obtain the prediction probabilities for the testing set
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