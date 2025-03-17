import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load Data
df = pd.read_csv('./SRSavg.csv')
feature_columns = ['SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION', 'SRS_COMMUNICATION', 'SRS_MOTIVATION', 'SRS_MANNERISMS']
target_column = 'HAS ADHD'

X = df[feature_columns]
y = df[target_column]


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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

with open("svm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)