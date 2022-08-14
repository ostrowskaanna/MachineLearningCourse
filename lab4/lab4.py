#!/usr/bin/env python
# coding: utf-8

# In[476]:


from sklearn import datasets


# In[477]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[478]:


data_breast_cancer.data


# In[479]:


data_breast_cancer.target


# In[480]:


data_breast_cancer.frame


# In[481]:


from sklearn.model_selection import train_test_split
X = data_breast_cancer.data
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[482]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[483]:


X_train[['mean area', 'mean smoothness']]


# In[484]:


svm_clf = Pipeline([('linear_svc', LinearSVC(C=1, loss='hinge'))])
svm_clf.fit(X_train[['mean area', 'mean smoothness']], y_train)


# In[485]:


svm_clf.predict(X_test[['mean area', 'mean smoothness']])


# In[486]:


svm_clf_with_scaler = Pipeline([('scaler',StandardScaler()), ('linear_svc', LinearSVC(C=1, loss='hinge'))])
svm_clf_with_scaler.fit(X_train[['mean area', 'mean smoothness']], y_train)


# In[487]:


svm_clf_with_scaler.predict(X_test[['mean area', 'mean smoothness']])


# In[488]:


from sklearn.metrics import accuracy_score
accuracy = []


# In[489]:


#accuracy dla klasyfikatora bez skalowania zbior uczacy
acc = accuracy_score(y_train, svm_clf.predict(X_train[['mean area', 'mean smoothness']]))
accuracy.append(acc)


# In[490]:


#accuracy dla klasyfikatora bez skalowania zbior testowy
acc = accuracy_score(y_test, svm_clf.predict(X_test[['mean area', 'mean smoothness']]))
accuracy.append(acc)


# In[491]:


#accuracy dla klasyfikatora ze skalowaniem zbior uczacy
acc = accuracy_score(y_train, svm_clf_with_scaler.predict(X_train[['mean area', 'mean smoothness']]))
accuracy.append(acc)


# In[492]:


#accuracy dla klasyfikatora ze skalowaniem zbior testowy
acc = accuracy_score(y_test, svm_clf_with_scaler.predict(X_test[['mean area', 'mean smoothness']]))
accuracy.append(acc)


# In[493]:


accuracy


# In[494]:


import pickle 
with open('bc_acc.pkl', 'wb') as file:
    pickle.dump(accuracy, file)


# In[495]:


data_iris = datasets.load_iris(as_frame=True)


# In[496]:


X = data_iris.data
y = data_iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[497]:


X_train[['petal length (cm)','petal width (cm)']]


# In[498]:


svm_clf.fit(X_train[['petal length (cm)','petal width (cm)']], y_train)


# In[499]:


svm_clf.predict(X_test[['petal length (cm)','petal width (cm)']])


# In[500]:


svm_clf_with_scaler.fit(X_train[['petal length (cm)','petal width (cm)']], y_train)


# In[501]:


svm_clf_with_scaler.predict(X_test[['petal length (cm)','petal width (cm)']])


# In[502]:


accuracy = []


# In[503]:


#accuracy dla klasyfikatora bez skalowania zbior uczacy
acc = accuracy_score(y_train, svm_clf.predict(X_train[['petal length (cm)','petal width (cm)']]))
accuracy.append(acc)


# In[504]:


#accuracy dla klasyfikatora bez skalowania zbior testowy
acc = accuracy_score(y_test, svm_clf.predict(X_test[['petal length (cm)','petal width (cm)']]))
accuracy.append(acc)


# In[505]:


#accuracy dla klasyfikatora ze skalowaniem zbior uczacy
acc = accuracy_score(y_train, svm_clf_with_scaler.predict(X_train[['petal length (cm)','petal width (cm)']]))
accuracy.append(acc)


# In[506]:


#accuracy dla klasyfikatora ze skalowaniem zbior testowy
acc = accuracy_score(y_test, svm_clf_with_scaler.predict(X_test[['petal length (cm)','petal width (cm)']]))
accuracy.append(acc)


# In[507]:


accuracy


# In[508]:


with open('iris_acc.pkl', 'wb') as file:
    pickle.dump(accuracy, file)


# In[ ]:




