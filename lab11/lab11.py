#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# In[25]:


X_train.shape[1:]


# In[26]:


import keras
def build_model(n_hidden=1, n_neurons=25, optimizer="sgd", learning_rate=0.00001, momentum=0): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(X_train.shape[1:]))
    for i in range(0, n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))    
    model.add(tf.keras.layers.Dense(1))    
    if optimizer == "sgd":
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss="mse", metrics="mae")  
    elif optimizer == "nesterov":
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True), loss="mse", metrics="mae")
    elif optimizer == "momentum":
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), loss="mse", metrics="mae") 
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics="mae")
    return model


# In[43]:


import os
import time
import numpy as np 

early_stopping=tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.00)

root_logdir = os.path.join(os.curdir, "tb_logs")
def get_run_logdir(name): 
    run_id = name
    return os.path.join(root_logdir, run_id)


# In[44]:


import pickle 
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

learning_rates = [0.000001, 0.00001, 0.0001]
results=[]
for lr in learning_rates:
    name=""
    name += str(int(time.time()))
    name += "_lr_"
    name += str(lr)
    run_logdir = get_run_logdir(name)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    
    model = build_model(learning_rate=lr)
    model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, early_stopping])
    result = model.evaluate(X_test, y_test)
    mse = result[0]
    mae = result[1]
    results.append((lr, mse, mae))   
with open("lr.pkl", "wb") as file:
    pickle.dump(results, file)


# In[29]:


print(results) 


# In[45]:


results = []
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

for hl in range(0,4):
    name=""
    name += str(int(time.time()))
    name += "_hl_"
    name += str(hl)
    run_logdir = get_run_logdir(name)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    
    model = build_model(n_hidden=hl)
    model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, early_stopping])
    result = model.evaluate(X_test, y_test)
    mse = result[0]
    mae = result[1]
    results.append((hl, mse, mae)) 
with open("hl.pkl", "wb") as file:
    pickle.dump(results, file)


# In[31]:


print(results) 


# In[46]:


results = []
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

neurons_number = [5, 25, 125]
for nn in neurons_number:
    name=""
    name += str(int(time.time()))
    name += "_nn_"
    name += str(nn)
    run_logdir = get_run_logdir(name)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    
    model = build_model(n_neurons=nn)
    model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, early_stopping])
    result = model.evaluate(X_test, y_test)
    mse = result[0]
    mae = result[1]
    results.append((nn, mse, mae))
with open("nn.pkl", "wb") as file:
    pickle.dump(results, file)


# In[33]:


print(results) 


# In[47]:


results = []
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

optimizers = ["sgd", "nesterov", "momentum", "adam"]
for opt in optimizers:
    name=""
    name += str(int(time.time()))
    name += "_opt_"
    name += opt
    run_logdir = get_run_logdir(name)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    
    model = build_model(optimizer=opt, momentum=0.5)
    model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, early_stopping])
    result = model.evaluate(X_test, y_test)
    mse = result[0]
    mae = result[1]
    results.append((opt, mse, mae))
with open("opt.pkl", "wb") as file:
    pickle.dump(results, file)


# In[35]:


print(results) 


# In[48]:


results = []
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

momentums = [0.1, 0.5, 0.9]
for mom in momentums:
    name=""
    name += str(int(time.time()))
    name += "_mom_"
    name += str(mom)
    run_logdir = get_run_logdir(name)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    
    model = build_model(optimizer="momentum", momentum=mom)
    model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, early_stopping])
    result = model.evaluate(X_test, y_test)
    mse = result[0]
    mae = result[1]
    results.append((mom, mse, mae))  
with open("mom.pkl", "wb") as file:
    pickle.dump(results, file)    


# In[37]:


print(results) 


# In[38]:


param_distribs = {
"model__n_hidden": [0, 1, 2, 3],
"model__n_neurons": [5, 25, 125],
"model__learning_rate": [0.000001, 0.00001, 0.0001],
"model__optimizer": ["sgd", "nesterov", "momentum", "adam"],
"model__momentum": [0.1, 0.5, 0.9]
}


# In[39]:


import scikeras
from scikeras.wrappers import KerasRegressor
es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[40]:


from sklearn.model_selection import RandomizedSearchCV
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=30, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)
rnd_search_cv.best_params_


# In[41]:


with open("rnd_search.pkl", "wb") as file:
    pickle.dump(rnd_search_cv.best_params_, file)


# In[42]:


rnd_search_cv.best_params_


# In[ ]:




