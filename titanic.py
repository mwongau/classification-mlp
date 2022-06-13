# To find classification accuracy of survival of a passenger in Titanic dataset by MLP using 10-fold cross-validation

import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Titanic dataset is loaded by Seaborn.
# reference: https://seaborn.pydata.org/generated/seaborn.load_dataset.html 
titanic=sns.load_dataset('titanic')
print(titanic.info())
df = titanic.copy()
data= df.drop(columns=['sex', 'class', 'deck', 'embarked', 'embark_town', 'alone', 'who', 'adult_male', 'alive']) # use numerical features only
data = data.dropna() # remove rows with missing values 
print('length of dataset after pre-processing', len(data))
print(data.info())
print(data.describe())
labels = data.pop('survived')
          
model = MLPClassifier(max_iter=300, random_state=42, solver='lbfgs')
scores = cross_val_score(model, data, labels, cv=10) # 10-fold cross-validation
print("10-fold cross-validation accuracy= %.4f, std= %.4f" % (scores.mean(), scores.std()))
