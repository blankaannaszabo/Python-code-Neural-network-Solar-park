#!/usr/bin/env python
# coding: utf-8

# # Megfelelő könyvtárak importálása

# In[8]:


import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator


# In[9]:


filename='Hugyag_teljes_adatsor.csv.csv'
df=pd.read_csv('Hugyag_teljes_adatsor.csv', sep=';', encoding='latin-1')
print(df.info())


# In[10]:


import matplotlib.pyplot as plt


# In[12]:


df['Dátum']=pd.to_datetime(df['Dátum'])


# In[13]:


df.set_axis(df['Dátum'], inplace=True)


# # Korrelációs mátrix

# In[142]:


import seaborn as sns
corr_heat = df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr_heat, annot = True, square = True)


# # Bemeneti paraméter és célváltozó meghatározása

# In[33]:


Termeltenergia_adat=df['Termelt energia'].values


# In[34]:


Termeltenergia_adat=Termeltenergia_adat.reshape((-1,1))


# # Adatok szétválasztása

# In[134]:


split_percent=0.80


# In[135]:


split=int(split_percent*len(Termeltenergia_adat))


# In[136]:


Termeltenergia_tanítás=Termeltenergia_adat[:split]


# In[137]:


Termeltenergia_tesztelés=Termeltenergia_adat[split:]


# In[138]:


print(Termeltenergia_tanítás.shape, Termeltenergia_tesztelés.shape)


# In[139]:


Dátum_tanítás=df['Dátum'][:split]


# In[140]:


Dátum_teszt=df['Dátum'][split:]


# In[141]:


print(Dátum_tanítás.shape, Dátum_teszt.shape)


# # Modell felépítése

# In[89]:


look_back=15


# In[90]:


train_generator=TimeseriesGenerator(Termeltenergia_tanítás, Termeltenergia_tanítás, length=look_back, batch_size=20)
test_generator=TimeseriesGenerator(Termeltenergia_tesztelés, Termeltenergia_tesztelés, length=look_back, batch_size=1)


# In[91]:


from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[92]:


model=Sequential()


# In[93]:


model.add(
    LSTM(7,
        activation='relu',
        input_shape=(look_back, 1))
)

model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(1))
model.summary()


# In[94]:


from keras.utils.vis_utils import plot_model
plot_model(model, show_shapes = True, show_layer_names = True)


# In[95]:


model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])


# In[96]:


num_epochs=100
hist=model.fit(train_generator, epochs=num_epochs, verbose=1)


# In[97]:


plt.plot(hist.history['root_mean_squared_error'])
plt.title('Átlagos négyzetes hiba négyzetgyöke (RMSE)')
plt.xlabel('Epoch')
plt.ylabel('Hiba')
plt.show()


# In[98]:


plt.plot(hist.history['loss'])
plt.title('Átlagos négyzetes hiba')
plt.xlabel('Epoch')
plt.ylabel('Hiba')
plt.show()


# In[127]:


prediction=model.predict(test_generator)


# In[128]:


Termeltenergia_tanítás=Termeltenergia_tanítás.reshape((-1))
Termeltenergia_tesztelés=Termeltenergia_tesztelés.reshape((-1))
prediction=prediction.reshape((-1))


# In[121]:


import chart_studio.plotly as py
import plotly.graph_objs as go


# In[122]:


trace1=go.Scatter(
x=Dátum_tanítás,
y=Termeltenergia_tanítás,
mode='lines',
name='Tanító adathalmaz')


# In[123]:


trace2=go.Scatter(
x=Dátum_teszt,
y=prediction,
mode='lines',
name='Előrejelzés')


# In[124]:


trace3=go.Scatter(
x=Dátum_teszt,
y=Termeltenergia_tesztelés,
mode='lines',
name='Ground Truth')


# In[125]:


layout=go.Layout(
title="Napelempark által termelt energia",
xaxis={'title':"Dátum"},
yaxis={'title':"Termelt energia (kW)"}
)


# In[126]:


fig=go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

