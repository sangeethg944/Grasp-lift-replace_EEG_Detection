#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, hamming_loss
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from average_metrics import average_bal_acc_score, average_f1_score, average_acc_score, average_hamming_loss
import numpy as np
import pandas as pd

# In[ ]:


def SupportVectorMachine(x_train, x_test, y_train, y_test):
    cols = y_test.columns.tolist()
    
    clf = svm.SVC(max_iter = 100, class_weight="balanced")
    result = np.empty((len(y_test),len(y_test.columns)))

    labelcounter = 0    
    for col in cols:
        clf.fit(x_train, y_train[[col]].values.ravel())        
        predictions = clf.predict(x_test)
        result[:,labelcounter] = predictions

        labelcounter+=1

    y_pred = pd.DataFrame(columns=cols, data=result)
    
    print("The average balanced accuracy score of the support vector classifier is", average_bal_acc_score(y_test,y_pred))
    print("The average accuracy score of the support vector classifier is ", average_acc_score(y_test,y_pred))
    print("The average F1 score of the support vector classifier is ", average_f1_score(y_test,y_pred))
    print("The average hamming loss of the support vector classifier is ", average_hamming_loss(y_test,y_pred))
    
    return y_pred
    
# In[ ]:


def LogisticRegression(x_train, x_test, y_train, y_test):
    cols = y_test.columns.tolist()
    
    clf = LogisticRegression(solver='lbfgs', max_iter = 100, class_weight="balanced")
    result = np.empty((len(y_test),len(y_test.columns)))
    
    labelcounter = 0    
    for col in cols:
        clf.fit(x_train, y_train[[col]].values.ravel())        
        predictions = clf.predict(x_test)
        result[:,labelcounter] = predictions
        labelcounter+=1

    y_pred = pd.DataFrame(columns=cols, data=result)
    
    print("The average balanced accuracy score of the logistic regression classifier is", average_bal_acc_score(y_test,y_pred))
    print("The average accuracy score of the logistic regression classifier is ", average_acc_score(y_test,y_pred))
    print("The average F1 score of the logistic regression classifier is ", average_f1_score(y_test,y_pred))
    print("The average hamming loss of the logistic regression classifier is ", average_hamming_loss(y_test,y_pred))
    
    return y_pred



# In[ ]:


def RandomForest(no_of_estimators, x_train, x_test, y_train, y_test):
    cols = y_test.columns.tolist()
    
    clf = RandomForestClassifier(n_estimators = no_of_estimators, class_weight="balanced", random_state = 1)
    result = np.empty((len(y_test),len(y_test.columns)))
    
    labelcounter = 0    
    for col in cols:   
        clf.fit(x_train, y_train[[col]].values.ravel())        
        predictions = clf.predict(x_test)
        result[:,labelcounter] = predictions
        labelcounter+=1

    y_pred = pd.DataFrame(columns=cols, data=result)
    
    print("The average balanced accuracy score of the random forest classifier is ", average_bal_acc_score(y_test,y_pred))
    print("The average accuracy score of the random forest classifier is ", average_acc_score(y_test,y_pred))
    print("The average F1 score of the random forest classifier is ", average_f1_score(y_test,y_pred))
    print("The average hamming loss of the random forest classifier is ", average_hamming_loss(y_test,y_pred))
    
    return y_pred



# In[ ]:


def KNearestNeighbour(no_of_neighbors, x_train, x_test, y_train, y_test):   
    cols = y_test.columns.tolist()
    
    clf = KNeighborsClassifier(n_neighbors = no_of_neighbors, weights ="distance")
    result = np.empty((len(y_test),len(y_test.columns)))

    labelcounter = 0
    for col in cols:
        clf.fit(x_train, y_train[[col]].values.ravel())        
        predictions = clf.predict(x_test)
        result[:,labelcounter] = predictions
        labelcounter+=1
        
    y_pred = pd.DataFrame(columns=cols, data=result)

    print("The average balanced accuracy score of the K_neighbours classifier is ", average_bal_acc_score(y_test,y_pred))
    print("The average accuracy score of the K_neighbours classifier is ", average_acc_score(y_test,y_pred))
    print("The average F1 score of the K_neighbours classifier is ", average_f1_score(y_test,y_pred))
    print("The average hamming loss of the K_neighbours classifier is ", average_hamming_loss(y_test,y_pred))
    
    return y_pred


    
def AdaBoost(no_of_estimators, x_train, x_test, y_train, y_test):    
    cols = y_test.columns.tolist()
    
    clf = AdaBoostClassifier(n_estimators = no_of_estimators)
    result = np.empty((len(y_test),len(y_test.columns)))

    labelcounter = 0    
    for col in cols:
        clf.fit(x_train, y_train[[col]].values.ravel())        
        predictions = clf.predict(x_test)
        result[:,labelcounter] = predictions
        labelcounter+=1

    y_pred = pd.DataFrame(columns=cols, data=result)

    print("The average balanced accuracy score of the AdaBoost classifier is ", average_bal_acc_score(y_test,y_pred))
    print("The average accuracy score of the AdaBoost classifier is ", average_acc_score(y_test,y_pred))
    print("The average F1 score of the AdaBoost classifier is ", average_f1_score(y_test,y_pred))
    print("The average hamming loss of the AdaBoost classifier is ", average_hamming_loss(y_test,y_pred))
    
    return y_pred


def XGBoost(x_train, x_test, y_train, y_test):    
    cols = y_test.columns.tolist()
    
    clf = XGBClassifier()
    result = np.empty((len(y_test),len(y_test.columns)))

    labelcounter = 0    
    for col in cols:
        clf.fit(x_train, y_train[[col]].values.ravel())        
        predictions = clf.predict(x_test)
        result[:,labelcounter] = predictions
        labelcounter+=1

    y_pred = pd.DataFrame(columns=cols, data=result)

    print("The average balanced accuracy score of the XGBoost classifier is ", average_bal_acc_score(y_test,y_pred))
    print("The average accuracy score of the XGBoost classifier is ", average_acc_score(y_test,y_pred))
    print("The average F1 score of the XGBoost classifier is ", average_f1_score(y_test,y_pred))
    print("The average hamming loss of the XGBoost classifier is ", average_hamming_loss(y_test,y_pred))
    
    return y_pred
