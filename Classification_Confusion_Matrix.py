######## ALGORITHM ########

#Stage 1: Classification Model Accuracy trained on Anonymized Data 
#1. Build a classification model on the Anonymized Dataset
#2. Generate y-predictions with Original Data's X-values using Anonymized Dataset's classification model
#3. Generate a confusion matrix between Anonymized Dataset's y-predictions and Original Data's y-values
#Note: We acquire losses on Original Data because those are the real y-values. 
#      We want to see how closely Anonymized Model can perform on the real data.

#Stage 2: Classification Model Accuracy trained on Original Data 
#4. Build a classification model on the Original Dataset
#5. Generate y-predictions with Original Data's X-values using Original Dataset's classification model
#6. Generate a confusion matrix between Original Dataset's y-predictions and Original Data's y-values

#Stage 3: Classification Model Accuracy Comparison between training on Original Data and Anonymized Data
#7. Subtract confusion matrix of Anonymized Dataset with that of Original Dataset

# Testing Notes: 
# We will use the same testing dataset for Anonymized Dataset and Original Dataset. 
# We will also leave out same number of data randomly for training on Anonymized Dataset. 
# Anonymized Data preserves the size of the dataset in terms of the number of data points/rows.
# The Confusion Matrix differences may not be exactly reproducible due to the randomness of 
# selecting data for training and testing, and that the rows of the anonymized dataset do not match 
# the rows of the original data.

######## CODE ########

#### Import Packages ####
# RCF - One of the few models that can classify with both numerical and categorical features
from sklearn.ensemble import RandomForestClassifier 

# Other Packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import DataConversionWarning
from scipy.stats import chi2
import scipy.stats as st

#Ensemble variable for average Confusion matrix
n_ensemble = 100

#When to print to update ensemble, for convenience
n_update_ensemble_print = 50

#% threshold for Confusion Matrix Exclusion
n_threshold = 0

#Prediction and Confidence interval thresholds
n_interval_prediction = 90
n_interval_confidence = 95

#Suppress numpy notation 
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

#Make warnings shut up
warnings.filterwarnings("ignore")

#### Take in arguments ####
input = sys.argv[1]

#### Preprocessing ####
# Read Original CSV
data_original = pd.read_csv(input)
data_original = data_original.dropna()

# Read Anonymized CSV
pattern1 = input[:-4] + "_anonymized.csv"
data_anonymized = pd.read_csv(pattern1)
data_anonymized = data_anonymized.dropna()

#LabelEncoder on string datatypes
data_original_processed = pd.DataFrame()
data_anonymized_processed = pd.DataFrame()
le = LabelEncoder()
for (columnName, columnData) in data_original.iteritems():
    if(data_original.dtypes[columnName] == 'object'):
        le.fit(columnData.astype(str))
        data_original_processed[columnName] = le.transform(columnData.astype(str))
        for (columnName2, columnData2) in data_anonymized.iteritems():
            if (columnName2 == columnName):
                le.fit(columnData2.astype(str))
                data_anonymized_processed[columnName2] = le.transform(columnData2.astype(str))
    else:
        data_original_processed[columnName] = columnData
        for (columnName2, columnData2) in data_anonymized.iteritems():
            if (columnName2 == columnName):
                data_anonymized_processed[columnName2] = columnData2

#Pandas has a very crappy translation problem with outputting a only a few of the original floating points as NaNs in iteritems()   
data_original_processed = data_original_processed.dropna() 
data_anonymized_processed = data_anonymized_processed.dropna()

# Designate Input Original
inputDF_original = data_original_processed.loc[:, data_original.columns != 'output']

# Designate Output Original, labeled "output"
outputDF_original = data_original_processed[['output']]

# Designate Input Anonymized
inputDF_anonymized = data_anonymized_processed.loc[:, data_anonymized.columns != 'output']

# Designate Output Anonymized, labeled "output"
outputDF_anonymized = data_anonymized_processed[['output']]

#Correct_Data for prediction and confidence intervals
#Some may argue that we should take even the wrongly classified instances into account
#But I think in industry, people look for the correctly classified instances 
#Correct me if I am wrong here
Correct_Data = []

# Initialize confusion matrix sizes
y_anonymized_size = outputDF_anonymized['output'].nunique()
y_original_size = outputDF_original['output'].nunique()

# Matrix Exclusion Size
threshold = n_threshold/100
exclusion_list = []
y_original_counter = outputDF_original.groupby('output').size()
for (columnName, columnData) in y_original_counter.iteritems():
    min_values = threshold * np.sum(y_original_counter.values)
    if (columnData < min_values):
        y_anonymized_size -= 1
        y_original_size -= 1
        exclusion_list.append(columnName)

# Initialize confusion matrices
confusion_matrix_anonymized = []
confusion_matrix_original = []
unique_values = []

confusion_matrix_original = np.zeros([y_original_size, y_original_size])
confusion_matrix_anonymized = np.zeros([y_original_size, y_original_size])
column_values = outputDF_original[["output"]].values.ravel()
unique_values = pd.unique(column_values)
unique_values = unique_values.tolist()

# Matrix Exclusion
for c in range(0, len(exclusion_list)):
    unique_values.remove(exclusion_list[c])
    
#Start Ensemble of Confusion Matrices
for t in range(0, n_ensemble):
    temp_confusion_matrix_original = np.zeros([y_original_size, y_original_size])
    temp_confusion_matrix_anonymized = np.zeros([y_original_size, y_original_size])
    
    #Print ensemble number
    if ((t % n_update_ensemble_print) == 0):
        print("Ensemble #%d" % t)
        
    # Train-Test Split - Original Data
    X_original_train, X_original_test, y_original_train, y_original_test = train_test_split(inputDF_original, outputDF_original)

    # Only get Anonymized Data for training; Use the entire training set because there is less data altogether thanks to dropna(); 
    X_anonymized_train = inputDF_anonymized
    y_anonymized_train = outputDF_anonymized
    
    #### Classification Model on Anonymized Dataset ####
    # Intialize RCF
    clf_anonymized = RandomForestClassifier()

    # Build RCF model with anonymized Xs and ys
    clf_anonymized.fit(X_anonymized_train, y_anonymized_train)

    #### Generate y-predictions with Anonymized Dataset's model on Original Test Xs####
    y_pred_anonymized = clf_anonymized.predict(X_original_test)

    #### Generate Confusion Matrix with Original Test ys ####
    # Generate confusion matrix
    save_x = 0
    save_y = 0
    for i in range(0, len(y_pred_anonymized)):
        for j in range(0, len(unique_values)):
            if(y_original_test.values[i] == unique_values[j]):
                save_x = j
            if(y_pred_anonymized[i] == unique_values[j]):
                save_y = j
        temp_confusion_matrix_anonymized[save_x][save_y] += 1

    confusion_matrix_anonymized += temp_confusion_matrix_anonymized
    
    #### Classification Model on Original Dataset ####
    # Intialize RCF
    clf_original = RandomForestClassifier()

    # Build RCF model with original Xs and ys
    clf_original.fit(X_original_train, y_original_train)

    #### Generate y-predictions with Original Dataset's model on Original Test Xs####
    y_pred_original = clf_original.predict(X_original_test)

    save_x = 0
    save_y = 0
    for i in range(0, len(y_pred_original)):
        for j in range(0, len(unique_values)):
            if(y_original_test.values[i] == unique_values[j]):
                save_x = j
            if(y_pred_original[i] == unique_values[j]):
                save_y = j
        temp_confusion_matrix_original[save_x][save_y] += 1
    
    confusion_matrix_original += temp_confusion_matrix_original
    
    #Statistical Interval Calculations of % deviation from original dataset
    proportion_matrix_anonymized = np.copy(temp_confusion_matrix_anonymized)
    for u in range(0, np.shape(temp_confusion_matrix_anonymized)[0]):
        total_u_row = np.sum(temp_confusion_matrix_anonymized[u])
        proportion_matrix_anonymized[u] /= total_u_row
    
    proportion_matrix_original = np.copy(temp_confusion_matrix_original)
    for u in range(0, np.shape(temp_confusion_matrix_original)[0]):
        total_u_row = np.sum(temp_confusion_matrix_original[u])
        proportion_matrix_original[u] /= total_u_row
    
    for u in range(0, np.shape(proportion_matrix_original)[0]):
        Correct_Data.append(((proportion_matrix_anonymized[u][u] - proportion_matrix_original[u][u]) * 100))
    
#### Average Confusion matrices ####
confusion_matrix_anonymized /= n_ensemble
confusion_matrix_original /= n_ensemble

#### TESTS AND PRINTING ####
#### Print Confusion Matrices ####
print("\nConfusion Matrix with training on Original Data:\n", confusion_matrix_original)
print("\nConfusion Matrix with training on Anonymized Data:\n", confusion_matrix_anonymized)

#Average Number of Correctly Identified Classes
for p in range(0, np.shape(confusion_matrix_anonymized)[0]):
    print("\nAvg. No. Correct Class %d Identifications w/ Original Data: %.2f" % (p, (confusion_matrix_original[p][p])))
    print("Avg. No. Correct Class %d Identifications w/ Anonymized Data: %.2f" % (p, (confusion_matrix_anonymized[p][p])))
    print("Difference: %.2f" % ((confusion_matrix_anonymized[p][p] - confusion_matrix_original[p][p])))

""" 
THIS IS A BAD TEST DUE TO BAD ASSUMPTIONS
#### Two-Way χ^2 (Chi-Squared) Test ####

#Original Matrix is the Expected (E)
#Anonymized Matrix is the Observed (O)

#Set observed and expected
χ_squared_test_expected = confusion_matrix_original
χ_squared_test_observed = confusion_matrix_anonymized

#(O-E)^2/E
χ_squared_matrix = np.zeros([χ_squared_test_observed.shape[0], χ_squared_test_observed.shape[1]])
χ_squared = 0
for i in range(0, χ_squared_test_observed.shape[0]):
    for j in range(0, χ_squared_test_observed.shape[1]):
        χ_squared_matrix[i][j] = ((χ_squared_test_observed[i][j] - χ_squared_test_expected[i][j])**2)/χ_squared_test_expected[i][j]
        χ_squared += χ_squared_matrix[i][j]

p_value = chi2.pdf(χ_squared, χ_squared_test_observed.shape)[0]
print("\nChi-Squared = %.3f, P-value = %.3f" % (χ_squared, p_value))
"""

#### Print Proportional Confusion Matrices ####    
# Convert to Row Stochastic Matrices
for u in range(0, np.shape(confusion_matrix_anonymized)[0]):
    total_u_row = np.sum(confusion_matrix_anonymized[u])
    confusion_matrix_anonymized[u] /= total_u_row

for u in range(0, np.shape(confusion_matrix_original)[0]):
    total_u_row = np.sum(confusion_matrix_original[u])
    confusion_matrix_original[u] /= total_u_row
  
print("\nProportional Confusion Matrix with training on Original Data:\n", confusion_matrix_original)
print("\nProportional Confusion Matrix with training on Anonymized Data:\n", confusion_matrix_anonymized)
print() #Additional formatting

#### Proportional Confidence Interval -  Classes ####
#Separate CIs for each class
ci_dict = {}

for data in unique_values:
    ci_dict[data] = []
    
position = 0
for data in Correct_Data:
    key_value = unique_values[(position % len(unique_values))]
    ci_dict[key_value].append(data)
    position += 1

row = 0
for key in unique_values:
    lower_correct_data = st.t.interval((n_interval_confidence/100), len(ci_dict[key])-1, loc=np.mean(ci_dict[key]), scale=st.sem(ci_dict[key]))[0] 
    upper_correct_data = st.t.interval((n_interval_confidence/100), len(ci_dict[key])-1, loc=np.mean(ci_dict[key]), scale=st.sem(ci_dict[key]))[1] 
    print("%d-percent Confidence Interval of percent deviation from Original Dataset for Class %d: (%.2f, %.2f)" % (n_interval_confidence, row, lower_correct_data, upper_correct_data))
    row += 1

#### Proportional Prediction Interval ####
#Prediction Intervals have high variance anyway so I do not think anyone cares about this
if len(Correct_Data) > 0:
    lower_correct_data = np.quantile(Correct_Data, (1-(n_interval_prediction/100)))    
    upper_correct_data = np.quantile(Correct_Data, (n_interval_prediction/100))
    print("\n%d-percent Overall Prediction Interval of percent deviation from Original Dataset: (%.2f, %.2f)" % (n_interval_prediction, lower_correct_data, upper_correct_data))
    
#### Proportional Confidence Interval -  Overall ####
#Overall CI takes in correctly identified data for classes which have small number of data that have high deviation values thanks to ... well ... law of small numbers
#Thus, the overall CI averages over small-sized classes as large-sized classes to minimize variation due to small numbers
        
lower_correct_data = st.t.interval((n_interval_confidence/100), len(Correct_Data)-1, loc=np.mean(Correct_Data), scale=st.sem(Correct_Data))[0]
upper_correct_data = st.t.interval((n_interval_confidence/100), len(Correct_Data)-1, loc=np.mean(Correct_Data), scale=st.sem(Correct_Data))[1]
    
print("%d-percent Overall Confidence Interval of percent deviation from Original Dataset: (%.2f, %.2f)" % (n_interval_confidence, lower_correct_data, upper_correct_data))