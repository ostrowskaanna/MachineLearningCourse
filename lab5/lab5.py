#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target


# In[27]:


import numpy as np
import pandas as pd
size = 300
X_df = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y_df = w4*(X_df**4) + w3*(X_df**3) + w2*(X_df**2) + w1*X_df + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X_df, 'y': y_df})
df.plot.scatter(x='x',y='y')


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 

tree_clf = DecisionTreeClassifier(max_depth=1)
tree_clf.fit(X_train, y_train)
f1_train_max = f1_score(y_train, tree_clf.predict(X_train))
f1_test_max = f1_score(y_test, tree_clf.predict(X_test)) 
best_depth = 1
print("max_depth: ", 1)
print("f1 train: ", f1_train_max)
print("f1_test: ", f1_test_max)
print("\n")

for i in range(2, 10):
    tree_clf = DecisionTreeClassifier(max_depth=i)
    tree_clf.fit(X_train, y_train)
    f1_train = f1_score(y_train, tree_clf.predict(X_train))
    f1_test = f1_score(y_test, tree_clf.predict(X_test))
    
    if  f1_train >= f1_train_max and f1_test >= f1_test_max:
        best_depth = i
        f1_train_max = f1_train
        f1_test_max = f1_test   
    else:
        if abs(f1_train - f1_test) < abs(f1_train_max - f1_test_max):
            best_depth = i
            f1_train_max = f1_train
            f1_test_max = f1_test   
    
    print("max_depth: ", i)
    print("f1 train: ", f1_train)
    print("f1_test: ", f1_test)
    print("\n")


# In[30]:

best_depth = 3
tree_clf = DecisionTreeClassifier(max_depth=best_depth)
tree_clf.fit(X_train, y_train)
f1_train = f1_score(y_train, tree_clf.predict(X_train))
f1_test = f1_score(y_test, tree_clf.predict(X_test))
accuracy_train = accuracy_score(y_train, tree_clf.predict(X_train))
accuracy_test = accuracy_score(y_test, tree_clf.predict(X_test))
data = [best_depth, f1_train, f1_test, accuracy_train, accuracy_test]
print(data)


# In[31]:


import pickle 

with open('f1acc_tree.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[32]:


from sklearn.tree import export_graphviz

file = "bc.png"
export_graphviz(
    tree_clf,
    out_file=file,
    feature_names=["mean texture", "mean symmetry"],
    rounded=True,
    filled=True
)


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2)
X_df = X_df.reshape(-1, 1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[39]:


from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error as mse

tree_reg = DecisionTreeRegressor(max_depth=1)
tree_reg.fit(X_train, y_train)
mse_train_min = mse(y_train, tree_reg.predict(X_train))
mse_test_min = mse(y_test, tree_reg.predict(X_test))
best_depth = 1
print(1)
print(mse_train_min)
print(mse_test_min)
print("\n")

for i in range(2, 10):
    tree_reg = DecisionTreeRegressor(max_depth=i)
    tree_reg.fit(X_train, y_train)
    mse_train = mse(y_train, tree_reg.predict(X_train))
    mse_test = mse(y_test, tree_reg.predict(X_test))
    
    if mse_train < mse_train_min and mse_test < mse_test_min:
        mse_train_min = mse_train
        mse_test_min = mse_test
        best_depth = i
    else:
        if abs(mse_train - mse_test) < abs(mse_train_min - mse_test_min):
            mse_train_min = mse_train
            mse_test_min = mse_test
            best_depth = i
    print(i)
    print(mse_train)
    print(mse_test)
    print("\n")
print(best_depth)


# In[35]:


tree_reg = DecisionTreeRegressor(max_depth=best_depth)
tree_reg.fit(X_train, y_train)
mse_train = mse(y_train, tree_reg.predict(X_train))
mse_test = mse(y_test, tree_reg.predict(X_test))
data = [best_depth, mse_train, mse_test]
print(data)


# In[36]:


with open('mse_tree.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[37]:


file = "reg.png"
export_graphviz(
    tree_clf,
    out_file=file,
    feature_names=["x", "y"],
    rounded=True,
    filled=True
)

