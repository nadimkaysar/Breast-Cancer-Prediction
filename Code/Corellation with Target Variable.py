#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle, class_weight
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[4]:


init_notebook_mode(connected=True)
cf.go_offline()


# In[5]:


data = pd.read_csv('../Data/after_preprocess.csv')


# In[7]:


data.shape


# In[8]:


data.describe().T


# In[ ]:


#Feature correlations


# In[9]:


corr = data.corr(method = 'spearman')


# In[10]:


layout = cf.Layout(height=600,width=600)
corr.abs().iplot(kind = 'heatmap', layout=layout.to_plotly_json(), colorscale = 'RdBu')


# In[11]:


#Find highly correlated features


# In[12]:


new_corr = corr.abs()
new_corr.loc[:,:] = np.tril(new_corr, k=-1) # below main lower triangle of an array
new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)


# In[13]:


new_corr[new_corr.correlation > 0.4]


# In[ ]:


#Correlation with target variable


# In[16]:


corr_with_target = data.corrwith(data.diagnosis).sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index().head(20)
unique_values = data.nunique().to_frame('unique_values').reset_index()
corr_with_unique = pd.merge(corr_with_target, unique_values, on = 'index', how = 'inner')


# In[17]:


corr_with_unique


# In[21]:


sns.relplot(x="area_mean", y="radius_mean", hue="diagnosis",data=data);


# In[ ]:




