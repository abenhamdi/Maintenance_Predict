#!/usr/bin/env python
# coding: utf-8

# # TP Maintenance Prédictive

# ## Packages et data preparation

# In[1]:


from google.colab import drive
drive.mount("/content/drive", force_remount=True)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[3]:


plt.rcParams["figure.figsize"] = (12,4)


# In[4]:


get_ipython().system('unzip "/content/drive/My Drive/EPSI/data.zip" # lien à mettre à jour en fonction de la localisation de votre fichier sur votre drive')


# In[ ]:


# nom de tous les capteurs stockés dans une liste
sensor_list = []

for file in os.listdir():
    if file.endswith(".txt"):
        sensor_list.append(file)
        
sensor_list.remove('description.txt')
sensor_list.remove('documentation.txt')
sensor_list.remove('profile.txt')
print(sensor_list)
print(len(sensor_list))


# In[ ]:


# on stock ensuite les valeurs de tous les capteurs dans une liste de dataframe

df_list = []

for i in range(0,len(sensor_list)):
    df_list.append(pd.read_csv(sensor_list[i], sep = '\t', header = None))

len(df_list[0])


# In[ ]:


#un exemple d'un des dataframe
df_list[0].head()


# In[ ]:


# on cré un dataframe spécifique pour les états des équipements
df_fault = pd.read_csv('profile.txt', sep = '\t', header = None, names = ['Cooler','Valve','Pump','Accumulator','Stable'])
df_fault.head()


# ## EDA

# ### Analyse d'un cycle

# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=6)
fig.tight_layout(pad=2.0)

cycle = np.random.randint(0,len(df_list[0])) # we check a random cycle

print('Cooler in state {}'.format(df_fault['Cooler'].iloc[cycle]))
print('Valve in state {}'.format(df_fault['Valve'].iloc[cycle]))
print('Pump in state {}'.format(df_fault['Pump'].iloc[cycle]))
print('Accumulator in state {}'.format(df_fault['Accumulator'].iloc[cycle]))

for i in range(0,len(sensor_list)):
  row_chart = np.divmod(i,6)[0]
  col_chart = np.divmod(i,6)[1]
  ax[row_chart,col_chart].plot(df_list[i].iloc[cycle])
  ax[row_chart,col_chart].set_title(sensor_list[i][:-4])
  ax[row_chart,col_chart].set_ylim(-0.5,np.max(df_list[i].iloc[cycle])*1.1)

fig.delaxes(ax[2,5]) # remove last useless chart (17 sensors only)


# ### Superposition des tous les cycles d'un capteur

# In[ ]:


for i in range(0,len(df_list[sensor_list.index('FS1.txt')])):
  plt.plot(df_list[sensor_list.index('FS1.txt')].iloc[i])


# In[ ]:


for i in range(0,len(df_list[sensor_list.index('TS1.txt')])):
  plt.plot(df_list[sensor_list.index('TS1.txt')].iloc[i])


# In[ ]:


for i in range(0,len(df_list[sensor_list.index('PS1.txt')])):
  plt.plot(df_list[sensor_list.index('PS1.txt')].iloc[i])


# ### Analyse d'un capteur en particulier pour un équipement particulier

# In[ ]:


# il est intéressant de regarder les courbes d'un capteur selon les états d'un équipement: ici TS4 selon l'état du cooler

index_cool_3 = df_fault[df_fault['Cooler'] == 3].index
index_cool_20 = df_fault[df_fault['Cooler'] == 20].index
index_cool_100 = df_fault[df_fault['Cooler'] == 100].index

sensor_id = sensor_list.index('TS4.txt')

# on regarde 5 cas aléatoirement choisis

for i in range(0,5):

  fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(9,1))
  #fig.figsize = (2,2)

  ax1.plot(df_list[sensor_id].iloc[np.random.choice(index_cool_3)])
  ax2.plot(df_list[sensor_id].iloc[np.random.choice(index_cool_20)])
  ax3.plot(df_list[sensor_id].iloc[np.random.choice(index_cool_100)])

  ax1.set_title('Close to failure')
  ax2.set_title('Reduced efficiency')
  ax3.set_title('Normal')

  ax1.set_ylim([30,60])
  ax2.set_ylim([30,60])
  ax3.set_ylim([30,60])

plt.show()


# In[ ]:


max_temp = np.max(df_list[sensor_id],axis=1)

sns.boxplot(x = df_fault["Cooler"], y=max_temp, showmeans = True)


# In[ ]:


# il est intéressant de regarder les courbes d'un capteur selon les états d'un équipement: ici PS1 selon l'état de la valve

index_valv_100 = df_fault[df_fault['Valve'] == 100].index
index_valv_90 = df_fault[df_fault['Valve'] == 90].index
index_valv_80 = df_fault[df_fault['Valve'] == 80].index
index_valv_73 = df_fault[df_fault['Valve'] == 73].index

sensor_id = sensor_list.index('PS1.txt')

# on regarde 5 cas aléatoirement choisis
for i in range(0,5):
  fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(12,1))

  ax1.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_100)])
  ax2.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_90)])
  ax3.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_80)])
  ax4.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_73)])
  

  ax1.set_title('Optimal')
  ax2.set_title('Small lag')
  ax3.set_title('Severe lag')
  ax4.set_title('Close total failure')

  y_min = 100
  y_max = 200
  ax1.set_ylim([y_min,y_max])
  ax2.set_ylim([y_min,y_max])
  ax3.set_ylim([y_min,y_max])
  ax4.set_ylim([y_min,y_max])

plt.show()


# In[ ]:


# il est intéressant de regarder les courbes d'un capteur selon les états d'un équipement: ici PS1 selon l'état de la valve

index_valv_100 = df_fault[df_fault['Valve'] == 100].index
index_valv_90 = df_fault[df_fault['Valve'] == 90].index
index_valv_80 = df_fault[df_fault['Valve'] == 80].index
index_valv_73 = df_fault[df_fault['Valve'] == 73].index

sensor_id = sensor_list.index('PS1.txt')

# on regarde 5 cas aléatoirement choisis
for i in range(0,5):
  fig = plt.figure(figsize=(9,2))

  plt.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_100)][0:2000], color = 'green', label = 'optimal')
  plt.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_90)][0:2000], color = 'yellow', label = 'small lag')
  plt.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_80)][0:2000], color = 'orange', label = 'severe lag')
  plt.plot(df_list[sensor_id].iloc[np.random.choice(index_valv_73)][0:2000], color = 'red', label = 'close to total failure')

  y_min = 100
  y_max = 200
  plt.ylim([y_min,y_max])
  plt.legend()


plt.show()


# ### Analyse de corrélations

# In[ ]:


max_val = np.zeros((len(df_list[0]),len(sensor_list)))

for i in range(0,len(sensor_list)):
  max_val[:,i] = df_list[i].max(axis=1)


# In[ ]:


plt.figure(figsize=(16,9))
sns.heatmap(pd.concat((pd.DataFrame(max_val, columns = [(sensor[:-4] + ' max') for sensor in sensor_list]),df_fault),axis=1).corr(),annot=True,cmap=sns.color_palette("vlag", as_cmap=True),square=True, fmt=".1f")


# In[ ]:


std_val = np.zeros((len(df_list[0]),len(sensor_list)))

for i in range(0,len(sensor_list)):
  std_val[:,i] = df_list[i].std(axis=1)


# In[ ]:


plt.figure(figsize=(16,9))
sns.heatmap(pd.concat((pd.DataFrame(std_val, columns = [(sensor[:-4] + ' std') for sensor in sensor_list]),df_fault),axis=1).corr(),annot=True,cmap=sns.color_palette("vlag", as_cmap=True),square=True, fmt=".1f")


# In[ ]:


# on va chercher à prédire les états à partir des valeurs moyennes des capteurs
mean_val = np.zeros((len(df_list[0]),len(df_list)))

for i in range(0,len(df_list)):
    mean_val[:,i] = np.mean(df_list[i],axis=1)

plt.figure(figsize=(16,9))
sns.heatmap(pd.concat((pd.DataFrame(mean_val, columns = [(sensor[:-4] + ' mean') for sensor in sensor_list]),df_fault),axis=1).corr(),annot=True,cmap=sns.color_palette("vlag", as_cmap=True),square=True, fmt=".1f")    


# ## Modeling

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


# ### Decision trees with mean

# In[ ]:


SEED = 123

X_train, X_val, y_train, y_val = train_test_split(mean_val, df_fault, test_size=0.33, random_state=SEED)


# In[ ]:


dt = []

for equipment in ['Cooler', 'Valve', 'Pump', 'Accumulator']:
  dt.append(DecisionTreeClassifier(random_state= SEED, min_samples_leaf = 10).fit(X_train, y_train[equipment]))
  print(equipment + " :" + str(dt[-1].score(X_val,y_val[equipment])))
  plt.figure(figsize=(6,2))
  sns.heatmap(pd.DataFrame(confusion_matrix(y_val[equipment],dt[-1].predict(X_val)),columns=list(dt[-1].classes_),index=list(dt[-1].classes_)),annot=True, fmt='g')
  plt.show()


# In[ ]:


eqpt_list = ['Cooler', 'Valve', 'Pump', 'Accumulator']
eqpt = eqpt_list.index('Accumulator')

print("Model analysis for " + eqpt_list[eqpt])
print("Features importance:")
for name, importance in zip([(sensor[:-4] + ' mean') for sensor in sensor_list], dt[eqpt].feature_importances_):
    print(name, importance)

# check the depth of the tree
print("\n" + 'Depth of the tree: ' + str(dt[eqpt].get_depth()))

# check the number of nodes
print('Number of nodes: ' + str(dt[eqpt].tree_.node_count))

# check the number of leaves
print('Number of leaves: ' + str(dt[eqpt].get_n_leaves()))


# ### Decision trees with mean, max and std

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(np.concatenate((mean_val,std_val,max_val),axis=1), df_fault, test_size=0.2, random_state=SEED)


# In[ ]:


dt = []

for equipment in ['Cooler', 'Valve', 'Pump', 'Accumulator']:
  dt.append(DecisionTreeClassifier(random_state= SEED).fit(X_train, y_train[equipment]))
  print(equipment + " :" + str(dt[-1].score(X_val,y_val[equipment])))
  plt.figure(figsize=(6,2))
  sns.heatmap(pd.DataFrame(confusion_matrix(y_val[equipment],dt[-1].predict(X_val)),columns=list(dt[-1].classes_),index=list(dt[-1].classes_)),annot=True, fmt='g')
  plt.show()


# In[ ]:


dt[0].feature_importances_


# In[ ]:


eqpt_list = ['Cooler', 'Valve', 'Pump', 'Accumulator']
eqpt = eqpt_list.index('Valve')

features = [(sensor[:-4] + ' mean') for sensor in sensor_list] + [(sensor[:-4] + ' std') for sensor in sensor_list] + [(sensor[:-4] + ' max') for sensor in sensor_list]

print("Model analysis for " + eqpt_list[eqpt])
print("Features importance:")
for name, importance in zip(features, dt[eqpt].feature_importances_):
    print(name, importance)

# check the depth of the tree
print("\n" + 'Depth of the tree: ' + str(dt[eqpt].get_depth()))

# check the number of nodes
print('Number of nodes: ' + str(dt[eqpt].tree_.node_count))

# check the number of leaves
print('Number of leaves: ' + str(dt[eqpt].get_n_leaves()))


# ### Logistic regression with mean

# In[ ]:


SEED = 123


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()


# In[ ]:


mean_val_norm = min_max_scaler.fit_transform(mean_val)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(mean_val_norm, df_fault, test_size=0.2, random_state=SEED)


# In[ ]:


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

for equipment in ['Cooler', 'Valve', 'Pump', 'Accumulator']:
  log_reg = LogisticRegression(random_state= SEED).fit(X_train, y_train[equipment])
  print(equipment + " :" + str(log_reg.score(X_val,y_val[equipment])))
  plt.figure(figsize=(6,2))
  sns.heatmap(pd.DataFrame(confusion_matrix(y_val[equipment],log_reg.predict(X_val)),columns=list(log_reg.classes_),index=list(log_reg.classes_)),annot=True, fmt='g')
  plt.show()


# ### Logistic regression with mean, max and std

# In[ ]:


std_val_norm = min_max_scaler.fit_transform(std_val)
max_val_norm = min_max_scaler.fit_transform(max_val)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(np.concatenate((mean_val_norm,std_val_norm,max_val_norm),axis=1), df_fault, test_size=0.2, random_state=SEED)


# In[ ]:


for equipment in ['Cooler', 'Valve', 'Pump', 'Accumulator']:
  log_reg = LogisticRegression(random_state= SEED).fit(X_train, y_train[equipment])
  print(equipment + " :" + str(log_reg.score(X_val,y_val[equipment])))
  plt.figure(figsize=(6,2))
  sns.heatmap(pd.DataFrame(confusion_matrix(y_val[equipment],log_reg.predict(X_val)),columns=list(log_reg.classes_),index=list(log_reg.classes_)),annot=True, fmt='g')
  plt.show()


# ### MLP - valve

# In[ ]:


import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# In[ ]:


def mlp(dim_series,class_nb, seed, initializer):
    
    """
    Architecture basique d'un MPL
    """
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer,input_shape=(dim_series,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer = initializer),
        #tf.keras.layers.Dense(32, activation='relu', kernel_initializer = initializer),
        tf.keras.layers.Dense(class_nb, activation='softmax', kernel_initializer = initializer)
    ])    
    
    return model


# In[ ]:


X = pd.concat(df_list,axis = 1).to_numpy()


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)


# In[ ]:


label_encoder = LabelEncoder()
Y_valve = tf.one_hot(label_encoder.fit_transform(df_fault['Valve']), depth = len(df_fault['Valve'].unique())).numpy()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_norm, Y_valve, test_size=0.2, random_state=SEED)


# In[ ]:


LR = 0.0001
BATCH_SIZE = 128
EPOCHS = 50
CALLBACKS = None

# valve
initializer = tf.keras.initializers.GlorotNormal(seed=SEED)
adam = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False)
loss = tf.keras.losses.CategoricalCrossentropy()
model_valve = mlp(np.shape(X_train)[1],len(df_fault['Valve'].unique()),SEED, initializer)
model_valve.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
model_valve.summary()


# In[ ]:


history = model_valve.fit(X_train,y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data = (X_val,y_val), callbacks = CALLBACKS, verbose = 1)


# In[ ]:


# pour voir les courbes d'apprentissage
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# ### MLP - accumulator

# In[ ]:


label_encoder = LabelEncoder()
Y_acc = tf.one_hot(label_encoder.fit_transform(df_fault['Accumulator']), depth = len(df_fault['Accumulator'].unique())).numpy()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_norm, Y_acc, test_size=0.2, random_state=SEED)


# In[ ]:


LR = 0.0001
BATCH_SIZE = 128
EPOCHS = 300
CALLBACKS = None

# accumulator
initializer = tf.keras.initializers.GlorotNormal(seed=SEED)
adam = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False)
loss = tf.keras.losses.CategoricalCrossentropy()
model_acc = mlp(np.shape(X_train)[1],len(df_fault['Accumulator'].unique()),SEED, initializer)
model_acc.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
model_acc.summary()


# In[ ]:


history = model_acc.fit(X_train,y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data = (X_val,y_val), callbacks = CALLBACKS, verbose = 1)


# In[ ]:


# pour voir les courbes d'apprentissage
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

