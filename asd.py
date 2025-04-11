#!/usr/bin/env python
# coding: utf-8

# In[25]:


import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf




dataset = pd.read_csv('./asd/autism_child.csv')

dataset.head(100)




dataset.info()




dataset=dataset.drop(['Case_No'],axis=1)
dataset.head(100)




dataset.isnull().sum()
dataset = dataset.replace('?', np.nan)





dataset.isnull().sum()
dataset['relation'].unique()




dataset['Age_Mons'] = pd.to_numeric(dataset['Age_Mons'], errors='coerce')



dataset['Age_Mons'].isnull().sum()




dataset['Ethnicity'].fillna("Others", inplace=True)
dataset['relation'].fillna("Not Mentioned", inplace=True)


#


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
dataset['Sex'] = le1.fit_transform(dataset['Sex'])
dataset['Jaundice'] = le1.fit_transform(dataset['Jaundice'])
dataset['Ethnicity'] = le1.fit_transform(dataset['Ethnicity'])
dataset['Family_mem_with_ASD'] = le1.fit_transform(dataset['Family_mem_with_ASD'])
# Removed 'contry_of_res' as it does not exist in the dataset
#dataset['used_app_before'] = le1.fit_transform(dataset['used_app_before'])
dataset['relation'] = le1.fit_transform(dataset['relation'])
dataset['Class/ASD'] = le1.fit_transform(dataset['Class/ASD'])




dataset.reset_index(drop=True,inplace=True)
dataset.head()



X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(len(X))




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)









from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(svclassifier, X, y, cv=5, scoring=scoring)

print("Cross-validation results:")
for key in scores:
    print(f"{key}: {scores[key]}")


joblib.dump(svclassifier,'asd_model.pkl')