#!/usr/bin/env python
# coding=utf-8

# Author: Z.-L. Deng

# This script is used to build the classification model based on selected biomarkers
# by using Support Vector Machine with linear model.


"""
Building the classification model using SVM method.
"""

import numpy as np
from numpy import arange
from sklearn import preprocessing
from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, matthews_corrcoef


print(__doc__)

# For loading the data from tab delimited file
def read_data_table(inf):
    inputfile = inf

    datamatrix = np.array([map(float,line.split()[1:]) 
                            for line in open(inputfile).readlines()])

 
    label = np.array(map(int,[line.split()[0] 
                        for line in open(inputfile).readlines()]))

    dataform = {"data":datamatrix,"label":label}

    return(dataform)

if __name__=="__main__":

    # The selected 4 best biomarker candidates
    markers = [45, 68, 77, 63]

    # Loading the traning data set  from file
    top100 = read_data_table("top100.txt")
    # Loading the external test data set from file
    jorth100 = read_data_table("Jo_bwa_new_top100.txt")

    X,y = top100["data"][:,markers],top100["label"]
    Jx,Jy = jorth100["data"][:,markers],jorth100["label"]
    
    # Scaling the value in to range [0,1]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X)
    X_test = X_train
    Jx_test = min_max_scaler.fit_transform(Jx)
    y_train, y_test = y, y
    Jy_test = Jy

    # Grid search for best parameters
    tuned_parameters = [
                    {'kernel': ['linear'], 'C': [2**i for i in arange(-5,  7, 0.2)]}]

    # Precision and fi score are used as the benchmark

    scores = ['precision','f1']

    for score in scores:
        print("\n")
        print("#-----------------------------------------------------------------------------------------#")
        print("# Tuning hyper-parameters for %s                                                  #" % score)
        print("#-----------------------------------------------------------------------------------------#")

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3, scoring=score)
        
        clf.fit(X_train, y_train)

        print("Best parameters set found on data set:")
        print("------------------------------------------------------------------------------------------")
        print(clf.best_estimator_)
        print("\n")
        print("------------------------------------------------------------------------------------------")


        print("Detailed classification report:")
        print("------------------------------------------------------------------------------------------")
        print("The model is trained on the full data set.")
        print("The scores are computed on the full evaluation set.")
        print("------------------------------------------------------------------------------------------")

        #--prediction
        y_true, y_pred = y_test, clf.predict(X_test)

        Jy_true,Jy_pred = Jy_test,clf.predict(Jx_test)
        
        print("Training data set-------------------------------------------------------------------------")

        print(classification_report(y_true, y_pred))
        print("MCC: ")
        print(matthews_corrcoef(y_true,y_pred))


        print  y_true,"\n",y_pred
        print("#####--------------------------------------------------------------------------------#####")
        print("\n")

        print("External test data set--------------------------------------------------------------------")
        print(classification_report(Jy_true, Jy_pred))
        print("MCC: ")
        print(matthews_corrcoef(Jy_true,Jy_pred))
        print  Jy_true,"\n",Jy_pred
