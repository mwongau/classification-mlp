## Classification of survival for Titanic dataset

### Dataset used
The Titanic dataset is loaded as a Pandas DataFrame from Seaborn. 
Reference for Seaborn datasets:
https://seaborn.pydata.org/generated/seaborn.load_dataset.html

### Test condition
Only the numerical features of Titanic dataset are used in classification.
80% of the dataset is used as training data & 20% as test data.
Ten different partitions of the Titanic data into training & test data are used.
A multi-layer perceptron is used to determine the classification accuracy of survival on the 
test set for each partition.

### Test result
The average of the classification accuracies of the 10 different partitions is 70.2%
The accuracy might be further improved by using more features instead of using numerical features 
only. 
  
  
