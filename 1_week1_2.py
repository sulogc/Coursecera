import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
#print(tf.__version__)
#------------------------------------------------------------------------------------------
#multiple outputs 모델에 대해 araboza
# UCI의 Energy efficiency data set을 보자. 
# features: 8 , Labels:2 라서 multioutput model 쓰기 개꿀임.
# https://archive.ics.uci.edu/ml/datasets/Energy+efficiency

def format_output(d):
    y1 = d.pop('Y1')
    y1 = np.array(y1)
    y2 = d.pop('Y2')
    y2 = np.array(y2)
    return y1, y2

def plot_diff(y_ture, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())


url = './data/ENB2012_data.xlsx'
df = pd.read_excel(url)
df = df.sample(frac=1).reset_index(drop=True)
#print(df.head(10))

train, test = train_test_split(df, test_size = 0.2)
train_stats = train.describe()

train_stats.pop('Y1')
train_stats.pop('Y2')

train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)

norm_train_X = (train - train_stats['mean']) / train_stats['std']
norm_test_X = (test - train_stats['mean']) / train_stats['std']
#------------------------------------------------------------------------------------------
#Build Model

input_layer = Input(shape=(len(train.columns), ))
f_dense = Dense(units= '128', activation ='relu')(input_layer)
s_dense = Dense(units= '128', activation ='relu')(f_dense)

y1_output = Dense(units= '1', name='y1_output')(s_dense)
t_dense = Dense(units= '64', activation ='relu')(s_dense)

y2_output = Dense(units= '1', name='y2_output')(t_dense)

model = Model(inputs= input_layer, outputs = [y1_output, y2_output])

#print(model.summary())
#----------------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer = optimizer,
                loss = {'y1_output': 'mse', 'y2_output': 'mse'},
                metrics = { 'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                            'y2_output': tf.keras.metrics.RootMeanSquaredError()})

history = model.fit(norm_test_X, train_Y,
                    epochs=500, batch_size = 10, validation_data=(norm_test_X, test_Y))

loss,Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y = test_Y)
print("Loss = {}, Y1_loss = {}, Y2_loss = {}, Y1_rmse = {}, Y2_rmse = {}".format(loss,Y1_loss, Y2_loss, Y1_rmse, Y2_rmse))

Y_pred = model.prediction(norm_test_X0)
pl