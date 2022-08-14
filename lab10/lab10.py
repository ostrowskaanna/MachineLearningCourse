#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[2]:


X_train = X_train/255
X_test = X_test/255


# In[3]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]


# In[4]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"))
model.add(tf.keras.layers.Dense(300, activation="relu", name="hidden1"))
model.add(tf.keras.layers.Dense(100, activation="relu", name="hidden2"))
model.add(tf.keras.layers.Dense(10, activation="softmax", name="output"))


# In[5]:


model(X_train)


# In[6]:


model.summary()


# In[7]:


#tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# In[8]:


model.compile(optimizer="sgd", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="accuracy")


# In[9]:


import os
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[10]:


import numpy as np
from sklearn.model_selection import train_test_split
X_valid, X_rest, y_valid, y_rest = train_test_split(X_train, y_train, train_size=0.1)


# In[11]:


model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=tensorboard_cb)


# In[13]:


model.save('fashion_clf.h5')


# In[14]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()


# In[15]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[16]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[17]:


len(X_train[0])


# In[18]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation="relu", input_shape=[len(X_train[0])]))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer="sgd", loss="mse")


# In[19]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# In[20]:


import os
root_logdir = os.path.join(os.curdir, "housing_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[21]:


model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[22]:


model.save("reg_housing_1.h5")


# In[23]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation="relu", input_shape=[len(X_train[0])]))
model.add(tf.keras.layers.Dense(30, activation="relu"))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer="sgd", loss="mse")


# In[24]:


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[25]:


model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[26]:


model.save("reg_housing_2.h5")


# In[27]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(300, activation="relu", input_shape=[len(X_train[0])]))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer="sgd", loss="mse")


# In[28]:


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[29]:


model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])


# In[30]:


model.save("reg_housing_3.h5")


# In[ ]:




