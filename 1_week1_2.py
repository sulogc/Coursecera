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
print(df.describe())
