import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve,roc_auc_score, auc, precision_score, recall_score, f1_score,matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')

ds = pd.read_csv("data.csv")
print(ds.info())
# print(ds.head(5))

# Rows, cols
print("Records ",ds.shape[0], " Cols ",ds.shape[1])

# remove empty contents 
ds = ds.drop(['id','Unnamed: 32'],axis=1)
ds.isna().sum()
ds.dropna()


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# Features and target
X = ds.drop(columns=["diagnosis"])

y = ds["diagnosis"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   

def do_evaluation_metrics(y_test, y_pred):
    # Dictionary 
    metrics = {}

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    #disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm,
    #     display_labels=["Malignant", "Benign"]
    # )

    # disp.plot(cmap="Blues")
    # plt.show()

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_test,y_pred)
    # AUC Score
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)
    metrics["auc_score"] =  roc_auc_score(y_test,y_pred)
    # auc(y_test, y_pred)
    # Precision
    metrics["precision"] = precision_score(y_test,y_pred, average='binary')
    # Recall
    metrics["recall"] = recall_score(y_test,y_pred, average='binary')
    #f1 score
    # metrics["f1_score_result"] = f1_score(y_test,y_pred,average='weighted')
    metrics["f1_score_result"] = f1_score(y_test,y_pred)
    # MCC Score
    metrics["mcc_score"] = matthews_corrcoef(y_test, y_pred)
    
    return metrics

def do_print_metrics(dict):
    print(f"Accuracy    : {dict["accuracy"]:.4f}")
    print(f"AUC         : {dict["auc_score"]:.4f}")
    print(f"Precision   : {dict["precision"]:.4f}")
    print(f"Recall      : {dict["recall"]:.4f}")
    print(f"F1 Score    : {dict["f1_score_result"]:.4f}")
    print(f"MCC Score   : {dict["mcc_score"]:.4f}")
    pass  

def build_logistics_regression_model(X_train_scaled,y_train,X_test,y_test):
    # Input params : 
    #  X_train and y_train - Train Data
    #  X_test and y_test - Test data
    
    # Model
    clf = LogisticRegression(max_iter=10000, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluation metrics
    return do_evaluation_metrics(y_test, y_pred)    

def build_decision_tree_classifier_model(X_train_scaled,y_train,X_test,y_test):
    # Input params : 
    #  X_train and y_train - Train Data
    #  X_test and y_test - Test data

    # Define parameter grid
    param_grid = {
        'max_depth': [3,5,7,10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1,2, 4,8]
    }

    model = DecisionTreeClassifier(  max_depth=None, random_state=42 )
    # GridSearch
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    # Predict    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # print("Best Parameters:", grid.best_params_)
    # print("Best Score:", grid.best_score_)
    # print(best_model.classes_)
    # Evaluation metrics
    result  = do_evaluation_metrics(y_test, y_pred)
    result['accuracy'] = accuracy
    result["auc_score"] =  roc_auc_score(y_test, y_prob)
    print(result)
    return result
    # model.fit(X_train_scaled, y_train)

    # # Predict
    # y_pred = model.predict(X_test_scaled)
    
    # # Evaluation metrics
    # return do_evaluation_metrics(y_test, y_pred)


def build_KNN_classifier_model(X_train_scaled,y_train,X_test,y_test):
    # Input params : 
    #  X_train and y_train - Train Data
    #  X_test and y_test - Test data

    # Train model
    model = KNeighborsClassifier(n_neighbors=3)    
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    return do_evaluation_metrics(y_test, y_pred)

def build_NBGaussion_model(X_train_scaled,y_train,X_test,y_test):
    # Input params : 
    #  X_train and y_train - Train Data
    #  X_test and y_test - Test data

    # Train model
    model = GaussianNB()    
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    return do_evaluation_metrics(y_test, y_pred)

def build_random_forest_classifier_model(X_train_scaled,y_train,X_test,y_test):
    # Input params : 
    #  X_train and y_train - Train Data
    #  X_test and y_test - Test data

    # Define parameter grid
    params_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    # GridSearch
    grid = GridSearchCV(
        estimator=model,
        param_grid=params_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    # Predict    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # print("Best Parameters:", grid.best_params_)
    # print("Best Score:", grid.best_score_)
    # print(best_model.classes_)
    # Evaluation metrics
    result  = do_evaluation_metrics(y_test, y_pred)
    result['accuracy'] = accuracy
    result["auc_score"] =  roc_auc_score(y_test, y_prob)
    print(result)
    return result

def build_XGBoost_model(X_train_scaled,y_train,X_test,y_test):
    # Input params : 
    #  X_train and y_train - Train Data
    #  X_test and y_test - Test data

    # Train model
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,        
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    return do_evaluation_metrics(y_test, y_pred)   

# Logistic Regression
dict_lg_score = build_logistics_regression_model(X_train_scaled, y_train, X_test, y_test)
dict_lg_score['y_test'] = y_test
print("Logistics Regression Model ")
result = { }
result['LG'] = dict_lg_score

do_print_metrics(result['LG'])

# Decision Tree
dict_dt_score = build_decision_tree_classifier_model(X_train_scaled, y_train, X_test, y_test)
print("Decision Tree Classifier Forest Model ")
do_print_metrics(dict_dt_score)

# KNN Classifier
dict_knn_score = build_KNN_classifier_model(X_train_scaled, y_train, X_test, y_test)
print("KNN Classifier Model ")
do_print_metrics(dict_knn_score)

# Naive Bayes Gaussian Classifier
dict_nbg_score = build_NBGaussion_model(X_train_scaled, y_train, X_test, y_test)
print("Naive Bayes Gaussian Model ")
do_print_metrics(dict_nbg_score)

# Random Forest
dict_rf_score = build_random_forest_classifier_model(X_train_scaled, y_train, X_test, y_test)
print("Random Forest Model ")
do_print_metrics(dict_rf_score)

# XGBoost
dict_xgb_score = build_XGBoost_model(X_train_scaled, y_train, X_test, y_test)
print("XGBoost Model ")
do_print_metrics(dict_xgb_score)

