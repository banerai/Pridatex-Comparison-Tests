# Pridatex Comparison Tests
Classification and Regression Tests between Anonymized and Original data to see their closeness.

## Properly Reading in Data
1. In original CSV file, change name of the output column to 'output'
2. Generate anonymized CSV file from original CSV file.
3. Rename the anonymized file to "[original filename]_anonymized.csv".

## Classification Test

### Usage
> python [Classification_Confusion_Matrix.py] [original filename]

### Current Tests
1. Original and Anonymized Confusion Matrices
2. Average Number of Correct Class Identifications
3. ~~Two-Way Chi-Squared Test~~
4. Original and Anonymized Proportional Confusion Matrices
5. X-percent Confidence Interval of percent deviation from Original Dataset for each Class
6. Y-percent Overall Prediction Interval of percent deviation from Original Dataset
7. X-percent Overall Confidence Interval of percent deviation from Original Dataset 
