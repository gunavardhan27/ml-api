#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
subprocess.check_call(['pip', 'install', 'gradio'])
import gradio as gr
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.calibration import calibration_curve


# In[3]:


df = pd.read_csv('./SRSavg.csv')

df.info()


# In[4]:


feature_columns = ['SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION', 'SRS_COMMUNICATION', 'SRS_MOTIVATION', 'SRS_MANNERISMS']
target_column = 'HAS ADHD'

X = df[feature_columns]
y = df[target_column]

df.head()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


get_ipython().system('pip install catboost')
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)


# Initialize the CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,  # Number of trees
    depth=6,  # Tree depth
    learning_rate=0.1,  # Controls step size
    loss_function='Logloss',  # For binary classification
    eval_metric='Accuracy',
    verbose=100
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# In[7]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

# Define a larger search space for hyperparameters
param_grid = {
    'C': [1, 10,100],  # More values for regularization
    'gamma': [1, 0.1, 0.01],  # Wider range for RBF gamma
    'kernel': ['rbf'],  # Keeping the best performing kernel
}

# Initialize SVC
svm_model = SVC(probability=True)

# Using RandomizedSearchCV for better hyperparameter tuning
'''random_search = RandomizedSearchCV(
    svm_model, param_distributions=param_grid,
    n_iter=20, cv=10, verbose=2, random_state=42, n_jobs=-1
)'''
random_search = GridSearchCV(
    svm_model, param_grid,
    cv=5, verbose=2, n_jobs=-1
)
random_search.fit(X_train, y_train)

# Best Parameters
print(f"Best Parameters: {random_search.best_params_}")
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")

report = classification_report(y_test, y_pred)
print(report)

def predict_data(info):
    print(best_model.predict(info))


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Random Forest Performance:")
print(classification_report(y_test, rf_preds))
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")


# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)
log_reg_preds = log_reg_model.predict(X_test)

print("Logistic Regression Performance:")
print(classification_report(y_test, log_reg_preds))
print(f"Accuracy: {accuracy_score(y_test, log_reg_preds):.4f}")


# In[11]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_train)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm')
plt.title("PCA Visualization of Data")
plt.show()


# In[12]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define MLP Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(64, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer (Binary Classification)
])




# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define an improved MLP model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),  # Prevent overfitting
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=16, verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Improved Test Accuracy: {test_acc:.4f}")


# In[ ]:




