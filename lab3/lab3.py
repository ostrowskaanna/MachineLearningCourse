#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Przygotowanie danych
import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y= w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8 - 4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv', index=None)
df.plot.scatter(x='x', y='y')


# In[2]:


#Podzial zbioru danych
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[3]:


X_new = np.array([[0], [2]])
X = X.reshape(-1, 1)                  
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
print(y[0], y[2])


# In[4]:


#Regresja liniowa 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.predict(X_new))


# In[5]:


krotka = (lin_reg, None)
reg = [krotka]


# In[6]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, lin_reg.predict(X_test), color='red')


# In[7]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, lin_reg.predict(X_train))


# In[8]:


train_mse = np.array(mean_squared_error(y_train, lin_reg.predict(X_train)))


# In[9]:


mean_squared_error(y_test, lin_reg.predict(X_test))


# In[10]:


test_mse = np.array(mean_squared_error(y_test, lin_reg.predict(X_test)))


# In[11]:


#KNN k=3
import sklearn.neighbors
knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)
print(knn_3_reg.predict(X_new))


# In[12]:


krotka = (knn_3_reg, None)
reg.append(krotka)


# In[13]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, knn_3_reg.predict(X_test), color='red')


# In[14]:


mean_squared_error(y_train, knn_3_reg.predict(X_train))


# In[15]:


train_mse = np.append(train_mse, mean_squared_error(y_train, knn_3_reg.predict(X_train)))


# In[16]:


mean_squared_error(y_test, knn_3_reg.predict(X_test))


# In[17]:


test_mse = np.append(test_mse, mean_squared_error(y_test, knn_3_reg.predict(X_test)))


# In[18]:


#KNN k=5
import sklearn.neighbors
knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train, y_train)
print(knn_5_reg.predict(X_new))


# In[19]:


krotka = (knn_5_reg, None)
reg.append(krotka)


# In[20]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, knn_5_reg.predict(X_test), color='red')


# In[21]:


mean_squared_error(y_train, knn_5_reg.predict(X_train))


# In[22]:


train_mse = np.append(train_mse, mean_squared_error(y_train, knn_5_reg.predict(X_train)))


# In[23]:


mean_squared_error(y_test, knn_5_reg.predict(X_test))


# In[24]:


test_mse = np.append(test_mse, mean_squared_error(y_test, knn_5_reg.predict(X_test)))


# In[25]:


#Regresja wielomianowa degree=2
from sklearn.preprocessing import PolynomialFeatures 
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_poly, y_train)
print(poly_2_reg.predict(poly_features.fit_transform([[0], [2]])))


# In[26]:


krotka = (poly_2_reg, poly_features)
reg.append(krotka)


# In[27]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, poly_2_reg.predict(poly_features.fit_transform(X_test)), color='red')


# In[28]:


mean_squared_error(y_train, poly_2_reg.predict(poly_features.fit_transform(X_train)))


# In[29]:


train_mse = np.append(train_mse, mean_squared_error(y_train, poly_2_reg.predict(poly_features.fit_transform(X_train))))


# In[30]:


mean_squared_error(y_test, poly_2_reg.predict(poly_features.fit_transform(X_test)))


# In[31]:


test_mse = np.append(test_mse, mean_squared_error(y_test, poly_2_reg.predict(poly_features.fit_transform(X_test))))


# In[32]:


#Regresja wielomianowa degree=3
from sklearn.preprocessing import PolynomialFeatures 
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_poly, y_train)
print(poly_3_reg.predict(poly_features.fit_transform([[0], [2]])))


# In[33]:


krotka = (poly_3_reg, poly_features)
reg.append(krotka)


# In[34]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, poly_3_reg.predict(poly_features.fit_transform(X_test)), color='red')


# In[35]:


mean_squared_error(y_train, poly_3_reg.predict(poly_features.fit_transform(X_train)))


# In[36]:


train_mse = np.append(train_mse, mean_squared_error(y_train, poly_3_reg.predict(poly_features.fit_transform(X_train))))


# In[37]:


mean_squared_error(y_test, poly_3_reg.predict(poly_features.fit_transform(X_test)))


# In[38]:


test_mse = np.append(test_mse, mean_squared_error(y_test, poly_3_reg.predict(poly_features.fit_transform(X_test))))


# In[39]:


#Regresja wielomianowa degree=4
from sklearn.preprocessing import PolynomialFeatures 
poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_poly, y_train)
print(poly_4_reg.predict(poly_features.fit_transform([[0], [2]])))


# In[40]:


krotka = (poly_4_reg, poly_features)
reg.append(krotka)


# In[41]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, poly_4_reg.predict(poly_features.fit_transform(X_test)), color='red')


# In[42]:


mean_squared_error(y_train, poly_4_reg.predict(poly_features.fit_transform(X_train)))


# In[43]:


train_mse = np.append(train_mse, mean_squared_error(y_train, poly_4_reg.predict(poly_features.fit_transform(X_train))))


# In[44]:


mean_squared_error(y_test, poly_4_reg.predict(poly_features.fit_transform(X_test)))


# In[45]:


test_mse = np.append(test_mse, mean_squared_error(y_test, poly_4_reg.predict(poly_features.fit_transform(X_test))))


# In[46]:


#Regresja wielomianowa degree=5
from sklearn.preprocessing import PolynomialFeatures 
poly_features = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_poly, y_train)
print(poly_5_reg.predict(poly_features.fit_transform([[0], [2]])))


# In[47]:


krotka = (poly_5_reg, poly_features)
reg.append(krotka)


# In[48]:


#Wizualizacja danych testowych
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, poly_5_reg.predict(poly_features.fit_transform(X_test)), color='red')


# In[49]:


mean_squared_error(y_train, poly_5_reg.predict(poly_features.fit_transform(X_train)))


# In[50]:


train_mse = np.append(train_mse, mean_squared_error(y_train, poly_5_reg.predict(poly_features.fit_transform(X_train))))


# In[51]:


mean_squared_error(y_test, poly_5_reg.predict(poly_features.fit_transform(X_test)))


# In[52]:


test_mse = np.append(test_mse, mean_squared_error(y_test, poly_5_reg.predict(poly_features.fit_transform(X_test))))


# In[53]:


train_mse


# In[54]:


test_mse


# In[55]:


d = {'train_mse': train_mse, 'test_mse': test_mse}
df = pd.DataFrame(data=d, index=['lin_reg', 'knn_3_reg', 'knn_5_reg', 
                                 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg'])


# In[56]:


df


# In[57]:


import pickle
with open('mse.pkl', 'wb') as file:
    pickle.dump(df, file)


# In[63]:


with open('mse.pkl', 'rb') as file:
    print(pickle.load(file))


# In[58]:


reg


# In[59]:


with open('reg.pkl', 'wb') as file:
    pickle.dump(reg, file)


# In[62]:


with open('reg.pkl', 'rb') as file:
    print(pickle.load(file))


# In[ ]:




