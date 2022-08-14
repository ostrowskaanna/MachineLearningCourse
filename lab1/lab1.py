#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'
r = requests.get(url, allow_redirects=True)


# In[2]:


open('housing.tgz', 'wb').write(r.content)


# In[3]:


import tarfile

t = tarfile.open('housing.tgz')
t.extractall()
t.close()


# In[4]:


import gzip 
import shutil

with open('housing.csv', 'rb') as f_in:
    with gzip.open('housing.csv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[5]:


import os 

os.remove('housing.tgz')


# In[6]:


import pandas as pd

df = pd.read_csv('housing.csv.gz')
df.head()


# In[7]:


df.info()


# In[8]:


print(df['ocean_proximity'].dtypes)


# In[9]:


df['ocean_proximity'].value_counts()


# In[10]:


df['ocean_proximity'].describe()


# In[11]:


import matplotlib.pyplot as plt # potrzebne ze względu na argument cmap

df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[12]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[13]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[14]:


s = df.corr()["median_house_value"].sort_values(ascending=False)


# In[15]:


s.reset_index().rename(columns={"index":"atrybut", "median_house_value":"wspolczynnik_korelacji"}).to_csv('korelacja.csv', 
                                                                                                          index=False)


# In[16]:


import seaborn as sns

sns.pairplot(df)


# In[17]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)


# In[18]:


train_set.head()


# In[19]:


test_set.head()


# In[20]:


train_set.corr()["median_house_value"].sort_values(ascending=False)


# In[21]:


test_set.corr()["median_house_value"].sort_values(ascending=False)


# In[22]:


import pickle

with open('train_set.pkl', 'wb') as file:
    pickle.dump(train_set, file)
with open('test_set.pkl', 'wb') as file:
    pickle.dump(test_set, file)






