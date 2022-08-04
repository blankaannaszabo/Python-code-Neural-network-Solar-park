#!/usr/bin/env python
# coding: utf-8

# # Megfelelő könyvtárak importálása

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# # Adathalmaz behívása

# In[2]:


adat = pd.read_csv('Hugyag_teljes_adatsor_szűrt.csv', sep=';', encoding='latin-1')
adat.head(10)


# # Korrelációs mátrix

# In[3]:


corr_heat = adat.corr(method='pearson',min_periods=1)
plt.figure(figsize = (22,22))
sns.heatmap(corr_heat, annot = True, square = True)


# # Adatok szétválasztása bemeneti paraméterekre és célváltozóra (X, Y)

# In[5]:


x = adat.iloc[:, 0:-1].values
y = adat.iloc[:, -1].values
y = np.reshape(y, (-1,1))
print(x.shape, y.shape)


# # Adatok szétválasztása tanuló és teszt adathalmazra

# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("Tanuló halmaz: {} {} \nTeszt halmaz: {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))


# # Adathalmaz normalizálása

# In[8]:


from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_trainsc = x_sc.fit_transform(x_train)
x_testsc = x_sc.transform(x_test)
y_sc = StandardScaler()
y_trainsc = y_sc.fit_transform(y_train)    
y_testsc = y_sc.transform(y_test)
x_trainsc


# # Előrecsatolt neurális háló felépítése

# In[10]:


def create_ann (n_layers, n_activation, kernels):
  model = tf.keras.models.Sequential()
  for i, nodes in enumerate(n_layers):
    if i == 0:
      model.add(Dense(nodes, kernel_initializer = kernels, activation = n_activation, input_dim = x_train.shape[1]))
    else:
      model.add(Dense(nodes, activation = n_activation, kernel_initializer = kernels))  
  model.add(Dense(1))
  model.compile(loss = 'mse', 
                optimizer = 'adam',
                metrics = [tf.keras.metrics.RootMeanSquaredError()])
  return model

solcast = create_ann([32, 64], 'relu', 'normal')
solcast.summary()


# In[12]:


from keras.utils.vis_utils import plot_model
plot_model(solcast, show_shapes = True, show_layer_names = True)


# In[13]:


hist = solcast.fit(x_train, y_train, batch_size = 32, validation_data = (x_test, y_test), epochs = 100, verbose = 2)


# # RMSE és MSE hiba datok megjelenítése

# In[14]:


plt.plot(hist.history['root_mean_squared_error'])
plt.title('Átlagos négyzetes hiba négyzetgyöke (RMSE)')
plt.xlabel('Epoch')
plt.ylabel('Hiba')
plt.show()


# In[15]:


plt.plot(hist.history['loss'])
plt.title('Átlagos négyzetes hiba')
plt.xlabel('Epoch')
plt.ylabel('Hiba')
plt.show()


# In[16]:


solcast.evaluate(x_train, y_train)


# # Előrejelzés és annak hibája

# In[17]:


from sklearn.metrics import mean_squared_error
y_pred = solcast.predict(x_test) 
y_pred_orig = y_sc.inverse_transform(y_pred) 
y_test_orig = y_sc.inverse_transform(y_test) 
RMSE_orig = mean_squared_error(y_pred_orig, y_test_orig, squared=False)
RMSE_orig


# In[18]:


train_pred = solcast.predict(x_train) 
train_pred_orig = y_sc.inverse_transform(train_pred) 
y_train_orig = y_sc.inverse_transform(y_train) 
mean_squared_error(train_pred_orig, y_train_orig, squared = False)


# In[19]:


from sklearn.metrics import r2_score
r2_score(y_pred_orig, y_test_orig)


# In[21]:


np.concatenate((train_pred_orig, y_train_orig), 1)


# In[22]:


np.concatenate((y_pred_orig, y_test_orig), 1)


# In[53]:


plt.figure(figsize = (16,6))
plt.subplot(1,2,2)
plt.scatter(y_pred_orig, y_test_orig)
plt.xlabel('Előrejelzett termelt energia')
plt.ylabel('Tényleges termelt energia')
plt.title('Teszt adathalmaz: Tényleges vs Előrejelzett termelt energiamennyiség')


# # Adatok vissza állítása a normalizált adathalmazból

# In[27]:


sc = StandardScaler()
pred_whole = solcast.predict(sc.fit_transform(x))
pred_whole_orig = y_sc.inverse_transform(pred_whole)
pred_whole_orig

