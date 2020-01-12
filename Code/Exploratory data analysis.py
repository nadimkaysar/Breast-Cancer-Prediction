#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data=pd.read_csv('../Data/after_preprocess.csv')


# In[6]:


data.shape


# In[7]:


data.head()


# In[8]:


data.describe()


# In[16]:


import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# In[17]:


M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]


# In[18]:


def plot_distribution(data_select, size_bin) :  
    tmp1 = M[data_select]
    tmp2 = B[data_select]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['malignant', 'benign']
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    
    fig['layout'].update(title = data_select)

    py.iplot(fig, filename = 'Density plot')


# In[20]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
plot_distribution('radius_mean', .5)


# In[21]:


plot_distribution('texture_mean', .5)


# In[22]:


plot_distribution('perimeter_mean', 5)


# In[23]:


plot_distribution('area_mean', 10)


# In[24]:


plot_distribution('radius_se', .1)
plot_distribution('texture_se', .1)
plot_distribution('perimeter_se', .5)
plot_distribution('area_se', 5)


# In[32]:


sns.distplot(data['radius_se'],kde = False, bins = 30)


# In[ ]:




