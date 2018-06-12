
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

# reading corresponding csv files 
dtrain = pd.read_csv("train.csv", encoding = 'ISO-8859-1')
dtest = pd.read_csv("test.csv")
addresses = pd.read_csv("addresses.csv")
lat_lon = pd.read_csv("latlons.csv")


# In[3]:

# merging addresses and ticket_id files to get the corresponding lon and lat for each ID
id_lan_lon = pd.merge(addresses, lat_lon, on='address')
id_lan_lon.head()


# In[4]:

# merging train data with id_lan_lon based on the ticket_id
dtrain = pd.merge(dtrain, id_lan_lon, on='ticket_id')
dtrain.set_index("ticket_id", inplace= True)
dtest = pd.merge(dtest, id_lan_lon, on='ticket_id')
dtest.set_index("ticket_id", inplace= True)
#dtrain.head()
#dtest.head()


# In[5]:

# Dropping rows which compliance is NaN
dtrain = dtrain[np.isfinite(dtrain['compliance'])]
dtrain = dtrain.reset_index(drop=True)
print(len(dtrain))
#dtrain


# In[6]:

feature_columns = ["fine_amount", "late_fee", "lat", "lon"]
label_column = ["compliance"]
train_df = dtrain[feature_columns + label_column]
train_df = train_df.dropna()
train_df = train_df.reset_index(drop=True)
#train_df
print(len(train_df))


# In[7]:

feature_columns =["fine_amount", "late_fee", "lat", "lon"]
test_df = dtest[feature_columns]
test_df = test_df.fillna(method = "ffill")
test_df.head(1)
#print(len(test_df))


# In[8]:

def blight_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    
    X_train = train_df[feature_columns]
    #print(X_train.shape)
    y_train = train_df[label_column].values.flatten()
    #print(y_train.shape)
    
    X_train_scaled = scaler.fit_transform(X_train)
    #print(X_train_scaled.shape)
    X_test_scaled = scaler.transform(test_df)
    #print(X_test_scaled.shape)

    
    nnclf = MLPClassifier(hidden_layer_sizes = [20, 15, 5], solver='lbfgs', alpha = 5.0, random_state = 0).fit(X_train_scaled, y_train)
    probs = nnclf.predict_proba(X_test_scaled)[:, 1]
   
    #test_df.set_index('ticket_id')
    
    #result = pd.Series(probs, index=test_df.index)
    
    
    #grid_values = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 4, 5]}
    #clf = GradientBoostingClassifier(random_state = 0)
    #grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
    #grid_clf_auc.fit(X_train, y_train)
    #probs = grid_clf_auc.predict_proba(test_df)[:, 1]
    result = pd.Series(probs, index=test_df.index)

    
    
    return result
blight_model()
    


# In[ ]:



