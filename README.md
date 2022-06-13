## Classification of survival of a passenger in Titanic dataset using MLP

### Dataset used
The Titanic dataset is loaded as a Pandas DataFrame by Seaborn. 
Reference for Seaborn datasets:
https://seaborn.pydata.org/generated/seaborn.load_dataset.html

### Test condition
Only the numerical features of Titanic dataset are used in classification.
10-fold cross-validation is used to determine the classification accuracy of survival.
The classifier used is multi-layer perceptron (MLP). 

### Test result
The 10-fold cross-validation accuracy is 72.00%.
The accuracy might be further improved by using more features instead of using numerical features 
only. 
  
  
