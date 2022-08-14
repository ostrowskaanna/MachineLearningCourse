#!/usr/bin/env python
# coding: utf-8

# In[115]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()
X_cancer = data_breast_cancer.data
y_cancer = data_breast_cancer.target


# In[116]:


from sklearn.datasets import load_iris 
data_iris = load_iris()
X_iris = data_iris.data
y_iris = data_iris.target 


# In[117]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)
pca.fit_transform(X_cancer)
print(pca.explained_variance_ratio_)


# In[118]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_cancer_scaled = scaler.fit_transform(X_cancer)
X2d = pca.fit_transform(X_cancer_scaled)
coeff = pca.explained_variance_ratio_
print(coeff)
components_cancer_scaled = pca.components_


# In[119]:


import pickle 
with open('pca_bc.pkl', 'wb') as file:
    pickle.dump(coeff, file)


# In[120]:


pca.fit_transform(X_iris)
print(pca.explained_variance_ratio_)


# In[121]:


X_iris_scaled = scaler.fit_transform(X_iris)
pca.fit_transform(X_iris_scaled)
coeff = pca.explained_variance_ratio_
print(coeff)
components_iris_scaled = pca.components_


# In[122]:


with open('pca_ir.pkl', 'wb') as file:
    pickle.dump(coeff, file)


# In[123]:


import numpy as np
idx_cancer = []
for comp in components_cancer_scaled:
    idx = np.argmax(abs(comp))
    idx_cancer.append(idx)
print(idx_cancer)    


# In[124]:


with open('idx_bc.pkl', 'wb') as file:
    pickle.dump(idx_cancer, file)


# In[125]:


idx_iris = []
for comp in components_iris_scaled:
    idx = np.argmax(abs(comp))
    idx_iris.append(idx)
print(idx_iris) 


# In[126]:


with open('idx_ir.pkl', 'wb') as file:
    pickle.dump(idx_iris, file)


# In[ ]:




