#!/usr/bin/env python
# coding: utf-8

# In[75]:


from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', version=1)


# In[76]:


mnist['data']


# In[77]:


mnist['target']


# In[78]:


import numpy as np
print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[79]:


import pandas as pd
X = pd.DataFrame(data = mnist['data'])
X


# In[80]:


y = pd.DataFrame(data = mnist['target'])
y


# In[81]:


y = y.sort_values(by = ['class'])
new_index = y.sort_values(by = ['class']).index
y


# In[82]:


X = X.reindex(index = new_index)
X


# In[83]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[84]:


print("y_train:", y_train.drop_duplicates().values.tolist())
print("y_test:", y_train.drop_duplicates().values.tolist())


# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[86]:


print("y_train:", y_train.drop_duplicates().values.tolist())
print("y_test:", y_train.drop_duplicates().values.tolist())


# In[87]:


y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')


# In[88]:


from sklearn.linear_model import SGDClassifier
from sklearn.utils.validation import column_or_1d
sgd_clf = SGDClassifier()
y_train_0 = column_or_1d(y_train_0)
sgd_clf.fit(X_train, y_train_0)


# In[89]:


print(sgd_clf.predict(X))


# In[90]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(score)


# In[91]:


import pickle 
with open('sgd_cva.pkl', 'wb') as file:
    pickle.dump(score, file)


# In[92]:


open_file = open('sgd_cva.pkl', 'rb')
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[93]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train_0, sgd_clf.predict(X_train))


# In[94]:


accuracy_score(y_test_0, sgd_clf.predict(X_test))


# In[95]:


accuracy = []
accuracy.append(accuracy_score(y_train_0, sgd_clf.predict(X_train)))
accuracy.append(accuracy_score(y_test_0, sgd_clf.predict(X_test)))


# In[96]:


print(accuracy)


# In[97]:


with open('sgd_acc.pkl', 'wb') as file:
    pickle.dump(accuracy, file)


# In[98]:


open_file = open('sgd_acc.pkl', 'rb')
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[99]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
sgd_m_clf = SGDClassifier()
y_train = column_or_1d(y_train)
sgd_m_clf.fit(X_train, y_train)


# In[100]:


print(sgd_m_clf.predict(mnist["data"]))


# In[101]:


score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
print(score)


# In[102]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)


# In[103]:


conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)


# In[104]:


with open('sgd_cmx.pkl', 'wb') as file:
    pickle.dump(conf_mx, file)


# In[105]:


open_file = open('sgd_cmx.pkl', 'rb')
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[ ]:




