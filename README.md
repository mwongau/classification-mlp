## Classification of survival of a passenger in Titanic dataset using MLP

### Dataset used
The Titanic dataset is loaded as a Pandas DataFrame by Seaborn. 
Reference for Seaborn datasets:
https://seaborn.pydata.org/generated/seaborn.load_dataset.html

### Test condition
The classifier used is multi-layer perceptron (MLP). 
The following features of Titanic dataset are used in classification: pclass, sex, age, sibsp, parch, fare.
The class label is "survived". 
Numerical values of the following features 'pclass', 'age', 'sibsp', 'parch', 'fare' are scaled to values between 0 and 1 by using MinMaxScaler of scikit-learn.
10-fold cross-validation is used to determine the classification accuracy of survival.

### Test result
The 10-fold cross-validation accuracy is 82.08%.
