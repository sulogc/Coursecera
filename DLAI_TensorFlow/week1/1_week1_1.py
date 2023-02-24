import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#print(tf.__version__)
#print(dir(tf.keras))
#------------------------------------------------------------------------------------------
# 일반적으로 쓰는 Sequential model
from tensorflow import keras
from tensorflow.keras import layers

seq_model = keras.Sequential([
    layers.Flatten(input_shape =(28, 28)),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(10, activation = 'softmax'),])
# Functional API를 통해서 만드려면, 
# 1. Explicitly define input layer (별개로 선언)
from tensorflow.keras.layers import Input
input = Input(shape=(28, 28))
# 2. Connecting each layer using Python functional systax 
# (이전 레이어를 파라미터, 현 레이어를 함수로 선언.)
from tensorflow.keras.layers import Dense, Flatten
# ...
x = Flatten()(input) #28-28 input을 Flatten해주면서 시작된다.
x = Dense(128, activation = 'relu')(x)      # 변수명을 구분해서 따로 사용할 수도 있다. 
predictions = Dense(10, activation = "softmax")(x)  #괄호를 통해 다음 함수로 연결된다.
# 시퀀셜 모델 처럼 레이어가 리스트 형태로 선언되지 않는다. 
#                                         
# 3. input, ouput layer를 주며 model 정의. 인풋은 다음과 같다.
from tensorflow.keras.models import Model
# ...
func_model = Model(inputs=input, outputs = predictions) #parameter 명이 복수형이다...
# func_model = Model(inputs=[input1, input2], outputs = [output1, output2]) 
# 인풋과 아웃풋을 특정하며 모델을 선언해준다. 
# 거의 비슷하지만, 시퀀셜 모델을 사용할 때와 달리 유연성있다.
# 괄호를 이중으로 쓰는게 직관적이지는 않지만, 사실 이건 파이썬 short cut 표현이다.

# #
# import tensorflow as tf
# first_dense = tf.keras.layers.Dense(128, activation = tf.nn.relu)(flatten_layer)
# #위 코드는 아래 코드의 숏컷이다. 
# first_dense = tf.keras.layers.Dense(128, activation = tf.nn.relu)
# first_dense(flatten_layer)
# first_dense 객체를 flatten_layer에 전달한다...고한다. 
#------------------------------------------------------------------------------------------


