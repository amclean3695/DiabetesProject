# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:00:29 2018

@author: Andrew McLean
Description: Code deployed on AWS Platform to perform a Comparison of Supervised Learning 
Algorithms in Predicting the Onset of Type 2 Diabetes. If ran in an IDE, classification metrics 
such as classification report, accuracy and confusion matrices are printed to console. The
metrics are included in the report. 
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fancyimpute as imp
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.model_selection import train_test_split

def classify(dict_classifiers, X_train, Y_train, X_test, Y_test):
    """ Batch classification of the classifiers listed in the dictionary and returns
        a dictionary of the model results
        Parameters
        ----------
        dict_classifiers : dictionary, includes the classifiers we use to fit the data
        X_train, Y_train, X_test, Y_test: pandas dataframes representing the data split into training 
        and testing sets
        Results
        ---------
        returns dictionary of the classifier results
        """
    
    #Specifies Labels for Classification Report
    target_names=['YES', 'NO'] 
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items()):
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        
        # Predictions of classifier
        train_pred = classifier.predict(X_train)
        test_pred = classifier.predict(X_test)
        
        # Accuracy scores
        train_score = metrics.accuracy_score(Y_train, train_pred)
        test_score = metrics.accuracy_score(Y_test, test_pred)
        
        # Classification Report
        cls_report = metrics.classification_report(Y_test, test_pred, target_names=target_names)
        
        #Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(Y_test, test_pred)
        
        dict_models[classifier_name] = {'Model': classifier, 'Train_score': train_score, 
                                        'Test_score': test_score, 'Train_time': t_diff,
                                        'Confusion Matrix': confusion_matrix,
                                        'Classification Report': cls_report 
                                        }
    return dict_models

def plot_confusion(cls, confusion_mat):
    """ Plots the confusion matrix for the specified classifier
        Parameters
        ----------
        cls : string, name of classifier for the plot
        confusion_mat: np.ndarry, the confusion matrix associated to the classifier
        Results
        ---------
        prints confusion matrix 
        """
    plt.matshow(confusion_mat)
    plt.title('Confusion Matrix for ' + cls)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def display_dict_models(dict_models, sort_by='Test_score'):
     """ Displays all of the metrics associated with training and testing the classifier
        Parameters
        ----------
        dict_models : dictionary, contains the results of model classifications
        sort_by: string, indicates how the dataframe should be sorted 
        Results
        ---------
        Prints dataframe of classifier results
        """
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['Test_score'] for key in cls]
    training_s = [dict_models[key]['Train_score'] for key in cls]
    training_t = [dict_models[key]['Train_time'] for key in cls]
    confusion_mat = [dict_models[key]['Confusion Matrix'] for key in cls]
    cls_report = [dict_models[key]['Classification Report'] for key in cls]
    
    result_df = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['Classifier', 'Train_score', 'Test_score', 'Train_time'])
    for ii in range(0,len(cls)):
        result_df.loc[ii, 'Classifier'] = cls[ii]
        result_df.loc[ii, 'Train_score'] = training_s[ii]
        result_df.loc[ii, 'Test_score'] = test_s[ii]
        result_df.loc[ii, 'Train_time'] = training_t[ii]
        plot_confusion(cls[ii], confusion_mat[ii])
        print('Classification Report for %s' % cls[ii])
        print(cls_report[ii])
    
    display(result_df.sort_values(by=sort_by, ascending=False))


# List containing names of the features
column_list = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
               'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv('Pimadiabetes.csv', names = column_list, header = None)

# Treat 0 in the biological variables other than number of times pregnant and outcome as missing values
replace_cols = [i for i in column_list if i not in ['Outcome','Pregnancies']]
df[replace_cols] = df[replace_cols].replace(0, np.nan)

# Median Imputation Technique for BMI, BloodPressure, Glucose
df[['BMI','BloodPressure','Glucose']] = df[['BMI','BloodPressure','Glucose']].fillna(df.median())

# Multiple Imputation Technique for SkinThickness and Insulin
df[['Insulin','SkinThickness']] = imp.MICE().complete(df[['Insulin','SkinThickness']]) 

# Splitting dataframe into predictors and outcome
Y_data = df['Outcome']
X_data = df.drop(['Outcome'], axis=1)

# Splits up the data set into 80% train 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = .2, random_state = 13)

# Dictionary of Classifiers 
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100,criterion='entropy')}  

# Displays the results of the classifiers
dict_models = classify(dict_classifiers, X_train, Y_train, X_test, Y_test)
display_dict_models(dict_models)