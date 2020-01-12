#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold

from sklearn.metrics import recall_score, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score,                             classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# In[2]:


data=pd.read_csv('../Data/after_preprocess.csv')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


#data Imbalance Check
data.diagnosis.value_counts(normalize=True)


# In[6]:


data.diagnosis.value_counts()


# In[7]:


data_major=data[data.diagnosis==0]


# In[8]:


data_minor=data[data.diagnosis==1]


# In[9]:


data_minor.head(1)


# In[10]:


data_minor_upsmapled = resample(data_minor, replace = True, n_samples = 357, random_state = 2018)


# In[37]:


data_minor_upsmapled.shape


# In[11]:


final_data=pd.concat([data_minor_upsmapled,data_major])


# In[12]:


final_data.diagnosis.value_counts()


# In[13]:


#Standarize the data


# In[14]:


X = final_data.drop('diagnosis', axis = 1)
Y = final_data.diagnosis


# In[15]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)


# In[16]:


mms = StandardScaler()
mms.fit(xtrain)
xtrain_scaled = mms.transform(xtrain)
xtest_scaled = mms.transform(xtest)


# In[ ]:


#logistic regression model


# In[17]:


logisticRegr = LogisticRegression()


# In[18]:


logisticRegr.fit(xtrain_scaled, ytrain)


# In[19]:


def evaluate_model(ytest, ypred, ypred_proba = None):
    if ypred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))
    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))


# In[20]:


lr_pred = logisticRegr.predict(xtest_scaled)


# In[21]:


evaluate_model(ytest, lr_pred)


# In[22]:


logisticRegr.score(xtest_scaled,ytest)


# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)


# In[25]:


classifier.fit(xtrain_scaled, ytrain)


# In[26]:


classifier.score(xtest_scaled,ytest)


# In[27]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[28]:


clf = SVC(kernel='linear')


# In[29]:


clf.fit(xtrain_scaled, ytrain)


# In[30]:


clf.score(xtest_scaled,ytest)


# In[ ]:




