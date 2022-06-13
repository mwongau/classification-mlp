# To classify survival of Titanic dataset by MLP

import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Titanic dataset is loaded by Seaborn.
# reference: https://seaborn.pydata.org/generated/seaborn.load_dataset.html 
titanic=sns.load_dataset('titanic')
print(titanic.info())
print(titanic.head())
df = titanic.copy()
data= df.drop(columns=['sex', 'class', 'deck', 'embarked', 'embark_town', 'alone', 'who', 'adult_male', 'alive'])
data = data.dropna() # remove rows with missing values 
print('length of dataset after pre-processing', len(data))
print(data.info())
labels = data.pop('survived')
train_data, test_data, y_train, y_test= train_test_split(data,labels,test_size=0.20,random_state=10)
          
model = MLPClassifier(max_iter=300)
model.fit(train_data, y_train)
score = model.score(test_data, y_test)
print('length of train & test data', len(train_data), len(test_data))
print('score', score)



