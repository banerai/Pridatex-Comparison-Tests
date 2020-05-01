######## ALGORITHM ########
#1. Forward Selection with Anonymized Data
#2. Forward Selection with Original Data
#3. Cross-Validation using same test data (Original Data)
#4. Compare MSE's

#Note: Forward Selection needs the following assumptions by ANOVA
#1. Data is Normally Distributed (may be covered wityh sample size > 30)
#2. Homogeneity of Variance - Variance among the groups should be approximately equal.
#3. Observations are independent of eachother (well, an inaccurate assumption we will take for this case)

######## CODE ########

#### Import Packages ####
import numpy as np 
import pandas as pd 
import os
import sys
import statsmodels.formula.api as sm
import statsmodels.sandbox.tools.cross_val as cross_val
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model as lm
from regressors import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut
import statsmodels.api as sma
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# RCF - One of the few models that can classify with both numerical and categorical features
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score as acc

#Boolean for whether forward or backward selection
bool_forward = False

#### Take in arguments ####
input = sys.argv[1]

#### Preprocessing ####
# Read Original CSV
data_original = pd.read_csv(input)
data_original.columns = data_original.columns.str.replace(' ', '_')
data_original.columns = data_original.columns.str.replace('(', '')
data_original.columns = data_original.columns.str.replace(')', '')
data_original.columns = data_original.columns.str.replace(':', '_')
data_original.columns = data_original.columns.str.replace('/', '_per_')
data_original.columns = data_original.columns.str.replace('-', '_')
data_original = data_original.dropna()

# Read Anonymized CSV
pattern1 = input[:-4] + "_anonymized.csv"
data_anonymized = pd.read_csv(pattern1)
data_anonymized.columns = data_anonymized.columns.str.replace(' ', '_')
data_anonymized.columns = data_anonymized.columns.str.replace('(', '')
data_anonymized.columns = data_anonymized.columns.str.replace(')', '')
data_anonymized.columns = data_anonymized.columns.str.replace(':', '_')
data_anonymized.columns = data_anonymized.columns.str.replace('/', '_per_')
data_anonymized.columns = data_anonymized.columns.str.replace('-', '_')
data_anonymized = data_anonymized.dropna()

# Designate Input Original
inputDF_original = data_original.loc[:, data_original.columns != 'output']

# Designate Output Original, labeled "output"
outputDF_original = data_original[['output']]

# Designate Input Anonymized
inputDF_anonymized = data_anonymized.loc[:, data_anonymized.columns != 'output']

# Designate Output Anonymized, labeled "output"
outputDF_anonymized = data_anonymized[['output']]

#Train Test Split
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(inputDF_original, outputDF_original)
X_train_anonymized, _, y_train_anonymized, __ = train_test_split(inputDF_anonymized, outputDF_anonymized)

y_train_original = y_train_original.values.ravel()
y_train_anonymized = y_train_anonymized.values.ravel()

#Fit Original Model
model_original = sfs(LinearRegression(),k_features=5,forward=bool_forward,verbose=2,cv=5,n_jobs=-1,scoring='r2')
model_original.fit(X_train_original,y_train_original)

feat_cols_original = list(model_original.k_feature_names_)

#Fit Anonymized Model
model_anonymized = sfs(LinearRegression(),k_features=5,forward=bool_forward,verbose=2,cv=5,n_jobs=-1,scoring='r2')
model_anonymized.fit(X_train_anonymized,y_train_anonymized)

feat_cols_anonymized = list(model_anonymized.k_feature_names_)

#Transform Original Data with respect to both models
inputDF_model_original = model_original.transform(inputDF_original)
inputDF_model_anonymized = model_anonymized.transform(inputDF_original)

#LOOCV
loocv = LeaveOneOut()
model = LinearRegression()

#Calculate MSEs with Original Data
rmse_loocv_original = np.sqrt(-cross_val_score(model, inputDF_model_original, outputDF_original, scoring="neg_mean_squared_error", cv = loocv))
rmse_loocv_anonymized = np.sqrt(-cross_val_score(model, inputDF_model_anonymized, outputDF_original, scoring="neg_mean_squared_error", cv = loocv))

mse_original = rmse_loocv_original.mean()
mse_anonymized = rmse_loocv_anonymized.mean()

#PRINTING
print("\n\nBest Original Predictors:\n", feat_cols_original)
print("Best Anonymized Predictors:\n", feat_cols_anonymized)

print("\nOriginal MSE: ", mse_original)
print("Anonymized MSE: ", mse_anonymized)

print("\nMSE Difference: ", (mse_anonymized - mse_original))
print("MSE Proportional Difference: ", (mse_anonymized - mse_original)/mse_original)
