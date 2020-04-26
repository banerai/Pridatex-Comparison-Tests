# Pridatex Comparison Tests
Classification and Regression Tests between Anonymized and Original data to see their closeness.

## Properly Reading in Data
1. In original CSV file, change name of the output column to 'output'
2. Generate anonymized CSV file from original CSV file.
3. Rename the anonymized file to "[original filename]_anonymized.csv".

## Classification Test

### Usage
> python [Classification_Confusion_Matrix.py] [original filename]

### Default Options
Ensemble variable for average Confusion matrix: n_ensemble = 100

When to print to update ensemble, for convenience: n_update_ensemble_print = 50

% threshold for Confusion Matrix Exclusion: n_threshold = 0

Prediction and Confidence interval thresholds: n_interval_prediction = 90, n_interval_confidence = 95

### Algorithm 

**Stage 1: Classification Model Accuracy trained on Anonymized Data**
1. Build a classification model on the Anonymized Dataset
2. Generate y-predictions with Original Data's X-values using Anonymized Dataset's classification model
3. Generate a confusion matrix between Anonymized Dataset's y-predictions and Original Data's y-values
*Note: We acquire losses on Original Data because those are the real y-values. 
       We want to see how closely Anonymized Model can perform on the real data.*

**Stage 2: Classification Model Accuracy trained on Original Data** 
4. Build a classification model on the Original Dataset
5. Generate y-predictions with Original Data's X-values using Original Dataset's classification model
6. Generate a confusion matrix between Original Dataset's y-predictions and Original Data's y-values

**Stage 3: Classification Model Accuracy Comparison between training on Original Data and Anonymized Data**
7. Subtract confusion matrix of Anonymized Dataset with that of Original Dataset

**Testing Notes**
We will use the same testing dataset for Anonymized Dataset and Original Dataset. 
We will also leave out same number of data randomly for training on Anonymized Dataset. 
Anonymized Data preserves the size of the dataset in terms of the number of data points/rows.
The Confusion Matrix differences may not be exactly reproducible due to the randomness of 
selecting data for training and testing, and that the rows of the anonymized dataset do not match 
the rows of the original data.


### Current Tests
1. Original and Anonymized Confusion Matrices
2. Average Number of Correct Class Identifications
3. ~~Two-Way Chi-Squared Test~~
4. Original and Anonymized Proportional Confusion Matrices
5. X-percent Confidence Interval of percent deviation from Original Dataset for each Class
6. Y-percent Overall Prediction Interval of percent deviation from Original Dataset
7. X-percent Overall Confidence Interval of percent deviation from Original Dataset 
