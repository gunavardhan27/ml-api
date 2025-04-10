import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

features = pd.read_csv('./Dyt-desktop.csv', delimiter=';').columns.values[:-1]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("./Dyt-desktop.csv", delimiter=';')  # adjust path
print(data['Dyslexia'].value_counts())



# Encode categorical columns using LabelEncoder (converts to 0/1 if binary)
label_cols = ['Gender']
le = LabelEncoder()

for col in label_cols:
    data[col] = le.fit_transform(data[col])
    
data['Nativelang'] = data['Nativelang'].map({'Yes': 1, 'No': 0})
data['Otherlang'] = data['Otherlang'].map({'Yes': 1, 'No': 0})


if data['Dyslexia'].dtype == object:
    data['Dyslexia'] = le.fit_transform(data['Dyslexia'])

y=data['Dyslexia']
X=data.drop(columns=['Dyslexia'])
# Standardize numerical columns
exclude_cols = [col for col in X.columns if col.startswith('Accuracy')]
cols_to_scale = [col for col in X.columns if col not in exclude_cols]

print(X.head(2))
print(y.head(3))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y,random_state=42)


ada = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_y_test = []
all_y_pred = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = CatBoostClassifier(
    iterations=1500,  # Number of trees
    depth=6,  # Tree depth
    learning_rate=0.03,  # Controls step size
    loss_function='Logloss',  # For binary classification
    eval_metric='F1',
    verbose=0,
    class_weights = {0: 1.0, 1:8.0 }
    #3252 / 392
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    y_pred = model.predict_proba(X_test)[:,1]
    threshold = 0.5
    y_pred_threshold = (y_pred >= threshold).astype(int)
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred_threshold)
    
print("Overall Accuracy:", accuracy_score(all_y_test, all_y_pred))
print(classification_report(all_y_test, all_y_pred))

joblib.dump(model,'updated_dyslexia.pkl')