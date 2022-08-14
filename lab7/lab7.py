#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml 
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score

scores = []

kmeans = KMeans(n_clusters=8)
labels = kmeans.fit_predict(X)
kmeans.predict(X)
score = silhouette_score(X, labels, metric='euclidean')
scores.append(score)

kmeans = KMeans(n_clusters=9)
labels = kmeans.fit_predict(X)
kmeans.predict(X)
score = silhouette_score(X, labels, metric='euclidean')
scores.append(score)

kmeans = KMeans(n_clusters=10)
labels10 = kmeans.fit_predict(X)
score = silhouette_score(X, labels10, metric='euclidean')
scores.append(score)

kmeans = KMeans(n_clusters=11)
labels = kmeans.fit_predict(X)
kmeans.predict(X)
score = silhouette_score(X, labels, metric='euclidean')
scores.append(score)

kmeans = KMeans(n_clusters=12)
labels = kmeans.fit_predict(X)
kmeans.predict(X)
score = silhouette_score(X, labels, metric='euclidean')
scores.append(score)


# In[3]:


print(scores)    


# In[4]:


import pickle
with open('kmeans_sil.pkl', 'wb') as file:
    pickle.dump(scores, file)


# In[5]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y, labels10)
print(conf_matrix)


# In[6]:


max_index = []
for row in conf_matrix:
    max_index.append(np.argmax(row))      
max_index.sort()
max_index = list(set(max_index))
print(max_index)


# In[7]:


with open('kmeans_argmax.pkl', 'wb') as file:
    pickle.dump(max_index, file)


# In[8]:


from numpy import linalg
euclidean = []
for i in range(0, 300):
    for j in range(0, len(X)):
        dist = linalg.norm(X[i]-X[j])
        if(dist > 0):
            euclidean.append(dist)


# In[9]:


euclidean.sort()


# In[10]:


distances = []
for i in range(0, 10):
    print(euclidean[i])
    distances.append(euclidean[i])


# In[11]:


with open('dist.pkl', 'wb') as file:
    pickle.dump(distances, file)


# In[12]:


s = (distances[0]+distances[1]+distances[2])/3
print(s)


# In[13]:


from sklearn.cluster import DBSCAN 

lengths = []

clustering = DBSCAN(eps = s)
clustering.fit_predict(X)
labels = []
for label in clustering.labels_:
    if label != -1:
        labels.append(label)
labels = list(set(labels))  
lengths.append(len(labels))

clustering = DBSCAN(eps = 1.04*s)
clustering.fit_predict(X)
labels = []
for label in clustering.labels_:
    if label != -1:
        labels.append(label)
labels = list(set(labels))  
lengths.append(len(labels))

clustering = DBSCAN(eps = 1.08*s)
clustering.fit_predict(X)
labels = []
for label in clustering.labels_:
    if label != -1:
        labels.append(label)
labels = list(set(labels))  
lengths.append(len(labels))


# In[14]:


with open('dbscan_len.pkl', 'wb') as file:
    pickle.dump(lengths, file)


# In[ ]:




