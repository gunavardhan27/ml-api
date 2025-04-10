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
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle



# Load Data
features = pd.read_csv('./Dyt-desktop.csv', delimiter=';').columns.values[:-1]

def load_data(path):
    data = pd.read_csv(path, delimiter=';')

    # Extract data points (features) as a NumPy array
    X = data.iloc[:, :-1].values

    # Extract labels as a NumPy array
    y = data['Dyslexia'].values
    y = np.where(y == 'Yes', 1, 0)

    return X, y

def pre_process(data, labels):
    X, y = data, labels

    # Replace NaN values with 0s
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = np.nan_to_num(X[i][j])
    

    # Encode 'Male' to 0 and 'Female' to 1
    X[:, 0] = np.where(X[:, 0] == 'Male', 0, 1)

    # Encode 'Yes' to 1 and 'No' to 0
    X[:, 1] = np.where(X[:, 1] == 'Yes', 1, 0)
    X[:, 2] = np.where(X[:, 2] == 'Yes', 1, 0)

    # Perform Min-Max scaling for non-'Accuracy' columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i, feature in enumerate(features):
        if not feature.startswith('Accuracy'):
            column_values = X[:, i].astype(float).reshape(-1, 1)
            X[:, i] = scaler.fit_transform(column_values).flatten()
    
    return X, y

def cross_validate(X, y, create_model, n_folds=10, threshold=0.5, seed=42, oversampling=None):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accuracies = []
    recalls = []
    precisions = []
    rocs = []
    f1_scores = []
    
    for train, test in kf.split(X):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        if oversampling == 'smote':
            oversampler = SMOTE(random_state=seed)
        if oversampling == 'adasyn':
            oversampler = ADASYN(random_state=seed)
        if oversampling is not None:
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
    return X_train,y_train,X_test,y_test

        # model = create_model(seed=seed)
        # model.fit(X_train, y_train)
        # return model
        # predicted_probabilities = model.predict_proba(X_test)
        # y_pred = (predicted_probabilities[:, 1] >= threshold).astype(int)

        # accuracy = metrics.accuracy_score(y_test, y_pred)
        # recall = metrics.recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        # precision = metrics.precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        # roc_score = metrics.roc_auc_score(y_test, y_pred)
        # f1_score = metrics.f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        
        # accuracies.append(accuracy)
        # recalls.append(recall)
        # precisions.append(precision)
        # rocs.append(roc_score)
        # f1_scores.append(f1_score)
    
    accuracy = float("{:.1f}".format(np.mean(accuracies) * 100))
    recall = float("{:.1f}".format(np.mean(recalls) * 100))
    precision = float("{:.1f}".format(np.mean(precisions) * 100))
    roc = float("{:.3f}".format(np.mean(rocs)))
    f1_score = float("{:.1f}".format(np.mean(f1_scores) * 100))

    return accuracy, recall, precision, roc, f1_score

def run_experiment(X, y, create_model=None, threshold=0.5, oversampling=None, seed=42):
    results = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'ROC', 'F1 Score', 'Threshold'])
    #accuracy, recall, precision, roc, f1_score = cross_validate(X, y, create_model, threshold=threshold, seed=seed, oversampling=oversampling)
    #results.loc[0] = [accuracy, recall, precision, roc, f1_score, "{:.3f}".format(threshold)]
    #return cross_validate(X, y, create_model, threshold=threshold, seed=seed, oversampling=oversampling)
    return cross_validate(X, y, create_model, threshold=threshold, seed=seed, oversampling=oversampling)

def catboost_algorithm(data,labels):
    X_train,y_train,X_test,y_test = cross_validate(data,labels,create_model=None,oversampling='adasyn',seed=42)
    model = CatBoostClassifier(
    iterations=1500,  # Number of trees
    depth=6,  # Tree depth
    learning_rate=0.03,  # Controls step size
    loss_function='Logloss',  # For binary classification
    eval_metric='Accuracy',
    verbose=100,)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class ("Yes")
    threshold = 0.35  # Adjust this based on precision-recall tuning
    y_pred_threshold = (y_prob >= threshold).astype(int)
    for index,i in enumerate(y_prob):
        if(i>=threshold):
            print('start',data[index],'end')
            break
    print("Accuracy:", accuracy_score(y_test, y_pred_threshold))
    print(classification_report(y_test, y_pred_threshold))
    return model

    # y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class ("Yes")
    # threshold = 0.35  # Adjust this based on precision-recall tuning
    # y_pred_threshold = (y_prob >= threshold).astype(int)
    # print("Accuracy:", accuracy_score(y_test, y_pred_threshold))
    # print(classification_report(y_test, y_pred_threshold))

def rf_200_balanced(seed=42):
    return RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)

#removed X,y, create_model from parameters
def rf_200_unbalanced(threshold=0.5, oversampling=None, seed=42):
    return RandomForestClassifier(n_estimators=200, class_weight=None, random_state=seed)

data, labels = load_data('./Dyt-desktop.csv')
data, labels = pre_process(data, labels)

my_model = catboost_algorithm(data,labels)

#result_exp_02 = run_experiment(data, labels, rf_200_unbalanced, oversampling=None, seed=42)


# Save Model
with open("svm_dyslexia_model.pkl", "wb") as f:
    pickle.dump(my_model, f)
