#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[4]:


data=pd.read_csv('../Data/data.csv')


# In[5]:


data.head()


# In[6]:


data.shape


# In[8]:


#Find the missing columns and their types


# In[9]:


df_dtypes = pd.merge(data.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         data.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[10]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# In[12]:


data.shape


# In[18]:


data = data.drop(['Unnamed: 32','id'],axis = 1)


# In[19]:


data.shape


# In[20]:


data.head()


# In[21]:


data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)


# In[25]:


data.head()


# In[ ]:


# Remove constant features


# In[26]:


def find_constant_features(dataFrame):
    const_features = []
    for column in list(dataFrame.columns):
        if dataFrame[column].unique().size < 2:
            const_features.append(column)
    return const_features


# In[27]:


const_features = find_constant_features(data)


# In[28]:


const_features


# In[29]:


#Remove Duplicate rows


# In[30]:


data.drop_duplicates(inplace=True)


# In[ ]:


# Removwe Duplicate Column


# In[31]:


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


# In[32]:


duplicate_cols = duplicate_columns(data)


# In[33]:


duplicate_cols


# In[34]:


data.to_csv('../Data/after_preprocess.csv', index = False)


# In[ ]:




