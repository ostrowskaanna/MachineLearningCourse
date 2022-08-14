#!/usr/bin/env python
# coding: utf-8

# In[333]:


from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# In[334]:


import pandas as pd
pd.concat([iris.data, iris.target], axis=1).plot.scatter(
x='petal length (cm)',
y='petal width (cm)',
c='target',
colormap='viridis'
)


# In[335]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[336]:


from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score
y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)
per_clf = Perceptron()
per_clf.fit(X_train, y_train_0)
a_tr = accuracy_score(y_train_0, per_clf.predict(X_train))
a_te = accuracy_score(y_test_0, per_clf.predict(X_test))
acc_0 = (a_tr, a_te)
print(acc_0)

w_0 = per_clf.intercept_[0]
w_1 = per_clf.coef_[0, 0]
w_2 = per_clf.coef_[0, 1]
wght_0 = (w_0, w_1, w_2) 
print(wght_0)


# In[337]:


y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)
per_clf = Perceptron()
per_clf.fit(X_train, y_train_1)
a_tr = accuracy_score(y_train_1, per_clf.predict(X_train))
a_te = accuracy_score(y_test_1, per_clf.predict(X_test))
acc_1 = (a_tr, a_te)
print(acc_1)

w_0 = per_clf.intercept_[0]
w_1 = per_clf.coef_[0, 0]
w_2 = per_clf.coef_[0, 1]
wght_1 = (w_0, w_1, w_2) 
print(wght_1)


# In[338]:


y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)
per_clf = Perceptron()
per_clf.fit(X_train, y_train_2)
a_tr = accuracy_score(y_train_2, per_clf.predict(X_train))
a_te = accuracy_score(y_test_2, per_clf.predict(X_test))
acc_2 = (a_tr, a_te)
print(acc_2)

w_0 = per_clf.intercept_[0]
w_1 = per_clf.coef_[0, 0]
w_2 = per_clf.coef_[0, 1]
wght_2 = (w_0, w_1, w_2) 
print(wght_2)


# In[339]:


acc_all = [acc_0, acc_1, acc_2]
print(acc_all)


# In[340]:


import pickle
with open('per_acc.pkl', 'wb') as file:
    pickle.dump(acc_all, file)


# In[341]:


wght_all = [wght_0, wght_1, wght_2]
print(wght_all)


# In[342]:


with open('per_wght.pkl', 'wb') as file:
    pickle.dump(wght_all, file)


# In[343]:


import numpy as np
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0,1,1,0])


# In[344]:


import tensorflow as tf
from tensorflow import keras


# In[345]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(2, input_dim = 2, activation="tanh"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD())


# In[346]:


history = model.fit(X, y, epochs=100, verbose=False)
print(history.history['loss'])


# In[347]:


model.predict(X)


# In[348]:


ok = False 
while not ok:
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_dim = 2, activation="tanh"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    history = model.fit(X, y, epochs=100, verbose=False)
    pred = model.predict(X)
    if(pred[0]<0.1 and pred[1]>0.9 and pred[2]>0.9 and pred[3]<0.1):
        ok = True
print(pred)        


# In[349]:


weights = model.get_weights()


# In[350]:


with open('mlp_xor_weights.pkl', 'wb') as file:
    pickle.dump(weights, file)

