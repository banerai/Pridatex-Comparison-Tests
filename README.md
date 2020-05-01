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
Ensemble variable for average Confusion matrix: n_ensemble = 1000

When to print to update ensemble, for convenience: n_update_ensemble_print = 50

% threshold for Confusion Matrix Exclusion: n_threshold = 0

Prediction and Confidence interval thresholds: n_interval_prediction = 90, n_interval_confidence = 95

### Algorithm 

![Classification Testing](/images/Classification_Testing_Algorithm.jpg)

**Classification Model Accuracy trained on Anonymized Data**
1. Build a classification model with the Anonymized Dataset
2. Generate y-predictions of Original Validation Data's X-values using Anonymized Dataset's classification model
3. Generate a confusion matrix between Anonymized Dataset's y-predictions and Original Validation Data's y-values
*Note: We acquire losses on Original Data because those are the real y-values. 
       We want to see how closely Anonymized Model can perform on the real data.*

**Classification Model Accuracy trained on Original Data** 
1. Build a classification model with the Original Training Dataset
2. Generate y-predictions of Original Validation Data's X-values using Original Training Dataset's classification model
3. Generate a confusion matrix between Original Training Dataset's y-predictions and Original Validation Data's y-values

### Current Tests
1. Original and Anonymized Confusion Matrices
2. Average Number of Correct Class Identifications
3. ~~Two-Way Chi-Squared Test~~
4. Original and Anonymized Proportional Confusion Matrices
5. X-percent Confidence Interval of percent deviation from Original Dataset for each Class
6. Y-percent Overall Prediction Interval of percent deviation from Original Dataset
7. X-percent Overall Confidence Interval of percent deviation from Original Dataset 

*Note: The intervals show a difference of accuracies between a model trained on the anonymized data and a model trained on the original data.*

**Testing Notes**

We will use the same validation dataset for Anonymized Dataset and Original Dataset. 
The Confusion Matrix differences may not be exactly reproducible due to the randomness of 
selecting data for training and testing, and that the rows of the anonymized dataset do not match 
the rows of the original data.
