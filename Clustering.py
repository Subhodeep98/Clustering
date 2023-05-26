#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


beer=pd.read_csv('beer.csv')
beer


# In[3]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_beer = scaler.fit_transform(beer[['CAL',
                                         'SOD',
                                         'ALC',
                                         'COST']])


# In[4]:


from sklearn.cluster import KMeans


# In[5]:


k = 3
clusters = KMeans(k, random_state = 42)
clusters.fit(scaled_beer)
beer["clusterid"] = clusters.labels_


# In[6]:


beer[beer.clusterid==0]


# In[7]:


beer[beer.clusterid==1]


# In[8]:


beer[beer.clusterid==2]


# In[9]:


from scipy.cluster.hierarchy import linkage, dendrogram


# In[10]:


complete_clustering = linkage(scaled_beer, method="complete", metric="euclidean")
average_clustering = linkage(scaled_beer, method="average", metric="euclidean")
single_clustering = linkage(scaled_beer, method="single", metric="euclidean")


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


dendrogram(complete_clustering)
plt.show()


# In[13]:


dendrogram(average_clustering)
plt.show()


# In[14]:


dendrogram(single_clustering)
plt.show()


# In[ ]:




