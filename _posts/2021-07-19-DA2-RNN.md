# 데이터 분석 심화 2 - RNN(Recurrent NN)

## A. RNN(순환신경망)

### 1. RNN(순환신경망)
- 순차데이터(sequence data)를 모델링하기 위한 신경망
- 음악, 동영상, 에세이, 시, 소스코드, 주가차트 등

### 2. 기존 신경망과의 차이점
- 은닉층에 순환에지(recurrent edge)가 존재
- t-1 시간에 발생한 정보를 t 시간으로 전달
- hidden state가 존재
- 이전 '정보'를 기억하는 역할

![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-07-19%20092658-2.png](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-07-19%20092658-2.png)

### 3. 순차 데이터
- 심전도 신호, 주식시세 차트, 음성, 문장, 유전자열 등
- 시간성 : 순서가 중요
- 가변길이 : 샘플마다 길이가 다름
- 문맥의존성(context dependency) : 어떤 요소들 사이에는 문맥 의존성이 있음
- 이미지 데이터에서 채널을 정해주듯 순차데이터에서도 분석에 맞게 처리해주어야 함

### 4. 텍스트 순차 데이터의 표현
- 단어가방(bag of words): 
  - 단어사전 각각의 문장 출현빈도를 카운트
  - 시간성 데이터의 유지가 어려우므로 RNN에는 부적합
- **원핫코드(one-hot code)**: 
  - 해당 단어의 위치(인덱스)만 1이고 나머지는 모두 0으로 표현
  - 단어 하나를 표현할 때에도 사전크기 만큼의 숫자가 사용됨
  - 서로 다른 단어 간의 유사도 측정기능이 없음
- 단어임베딩(word embedding): 
  - 단어 사이의 상호작용을 분석하여 새로운 공간으로 변환하는 기법
  - 학습을 통해 보통 사전의 크기보다 훨씬 작은 크기로 변환됨 (예) word2vec

## B. RNN의 이해

### 1. RNN의 구조
- 어떤 시점에서의 state가 다음 state를 계산하는 데 영향을 미치는 구조
- sequence data의 처리에 적합

![%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-07-19%20093558-2.png](attachment:%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-07-19%20093558-2.png)

### 2. RNN의 계산방법
- 출력(Y)는 필요한 어떤 순간 t에서 계산되어 출력됨
- vector x의 sequence의 모든 time step에 다음과 같은 순환공식을 적용함에 의해 계산
> **h_t = f_W(h_t−1, x_t)**
> - h_t : new state
> - f_W : some function with parameter W
> - h_t−1 : old state
> - x_t : input vector at some time step
- 모든 time step에서 **같은 함수(주로 tanh)와 같은 파라미터가 사용되어야 함**

### 3. RNN의 적용방법과 사례
1) one to one(Vanilla Neural Networks)
- RNN의 기본
- 매 입력마다 출력반응을 생성

2) one to many
- Image Captioning: image -> sequence of words, 사진이 입력되면 사진의 설명문장을 생성
- Language model

3) many to one
- Sentiment Classification(감성분류): seq. of words -> sentiment (예) 소비자의 상품평이 긍정적인지 부정적인지 판별
- 시계열 예측: 어떤 지역의 최근 날씨가 주어졌을 때 향후 날씨 예측

4) many to many
- Machine Translation: seq. of words -> seq. of words, 한국어 -> 영어 자동번역 등
- Video classification on frame level: (예) 블랙박스 영상 프레임으로 부터 사고 발생여부 판별
- Name entity recognition: (예) text에서 언급된 사람, 회사, 장소 등의 개체를 인식

## C. RNN의 구현
### 1. 2개의 sequence data만 있는 경우


```python
import numpy as np

# input data(mini batch)
# t = 0
X0_batch = np.array([[0, 1, 2],
                     [3, 4, 5],
                     [6, 7, 8],
                     [9, 0, 1]], dtype=np.float32)

# t = 1
X1_batch = np.array([[9, 8, 7],
                     [3, 4, 5],
                     [6, 5, 4],
                     [3, 2, 1]], dtype=np.float32)

print(f"X0_batch = {X0_batch.shape}")
print(f"X1_batch = {X1_batch.shape}")
```

    X0_batch = (4, 3)
    X1_batch = (4, 3)
    


```python
import numpy as np
import tensorflow as tf

# input data(mini batch)
hidden_size = 2

Wx = tf.Variable(tf.random.normal([3, hidden_size], dtype=tf.float32)) # 행 3개 필요(입력벡터 4*3)
Wy = tf.Variable(tf.random.normal([hidden_size, hidden_size], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, hidden_size], dtype=tf.float32))

@tf.function
def run(X0, X1):
    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)  # 기존 + X0 가중치곱
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)  # 기존 + X1 가중치곱 + 이전 hidden layer 가중치곱
    return Y0, Y1

_Y0, _Y1 = run(X0_batch, X1_batch)
print(f"Y0:{_Y0.shape}\n{_Y0}") # 입력 X0 hidden layer 
print(f"Y1:{_Y1.shape}\n{_Y1}") # 입력 X1 hidden layer
```

    Y0:(4, 2)
    [[ 0.99973756 -0.5601316 ]
     [ 1.          0.03313472]
     [ 1.          0.6039347 ]
     [ 1.         -0.9999998 ]]
    Y1:(4, 2)
    [[1.         0.9993193 ]
     [1.         0.6077035 ]
     [1.         0.96217805]
     [1.         0.99648744]]
    

### 2. Tensorflow 2.0 keras API 이용


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
```

1) 특정 Cell의 선언과 이를 루프화하는 코드를 사용


```python
# method 1: using Cell and Loop
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)
```

2) RNN API를 사용


```python
# method 2: using API
rnn = layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)
```

3) sequence_size가 1인 경우 : 'h'만 입력한 경우


```python
# one hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

sequence_length = 1
x_data = np.array([[h]], dtype=np.float32)

hidden_size = 2
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True)  # 전체 sequences, state RETURN
outputs, states = rnn(x_data)

print(f"x_data = {x_data.shape}\n{x_data}")
print(f"outputs = {outputs.shape}\n{outputs}")  # 전체 hidden state 값
print(f"states = {states.shape}\n{states}")  # 위와 같게 나옴
```

    x_data = (1, 1, 4)
    [[[1. 0. 0. 0.]]]
    outputs = (1, 1, 2)
    [[[ 0.61293894 -0.6207947 ]]]
    states = (1, 2)
    [[ 0.61293894 -0.6207947 ]]
    

4) sequence_size가 5인 경우 : 'hello'를 차례로 입력한 경우


```python
# one hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

sequence_length = 5
x_data = np.array([[h, e, l, l, o]], dtype=np.float32)

hidden_size = 2
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True) 
outputs, states = rnn(x_data)

print(f"x_data = {x_data.shape}\n{x_data}")
print(f"outputs = {outputs.shape}\n{outputs}") 
print(f"states = {states.shape}\n{states}")
```

    x_data = (1, 5, 4)
    [[[1. 0. 0. 0.]
      [0. 1. 0. 0.]
      [0. 0. 1. 0.]
      [0. 0. 1. 0.]
      [0. 0. 0. 1.]]]
    outputs = (1, 5, 2)
    [[[-0.4858402   0.6231402 ]
      [-0.5604765   0.32734135]
      [-0.2931562  -0.4979094 ]
      [-0.12564552 -0.88448596]
      [-0.69961536 -0.9126185 ]]]
    states = (1, 2)
    [[-0.69961536 -0.9126185 ]]
    

5) sequence_size 5, batch_size 3인 경우


```python
# one hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

sequence_length = 5
x_data = np.array([[h, e, l, l, o],
                   [e, l, l, o, h],
                   [l, l, o, h, e]], dtype=np.float32)

hidden_size = 2
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

print(f"x_data = {x_data.shape}\n{x_data}")
print(f"outputs = {outputs.shape}\n{outputs}") 
print(f"states = {states.shape}\n{states}")  
```

    x_data = (3, 5, 4)
    [[[1. 0. 0. 0.]
      [0. 1. 0. 0.]
      [0. 0. 1. 0.]
      [0. 0. 1. 0.]
      [0. 0. 0. 1.]]
    
     [[0. 1. 0. 0.]
      [0. 0. 1. 0.]
      [0. 0. 1. 0.]
      [0. 0. 0. 1.]
      [1. 0. 0. 0.]]
    
     [[0. 0. 1. 0.]
      [0. 0. 1. 0.]
      [0. 0. 0. 1.]
      [1. 0. 0. 0.]
      [0. 1. 0. 0.]]]
    outputs = (3, 5, 2)
    [[[-0.6126506  -0.06350043]
      [-0.86930865  0.86428136]
      [-0.8772982   0.29534802]
      [-0.74212784  0.60453814]
      [-0.9448845  -0.6292621 ]]
    
     [[-0.7426959   0.6753254 ]
      [-0.8142633   0.32906485]
      [-0.73362476  0.55892134]
      [-0.94054157 -0.61387175]
      [-0.7216307   0.77867275]]
    
     [[-0.13785852  0.2580866 ]
      [-0.3978935   0.1853213 ]
      [-0.84561884 -0.6065247 ]
      [-0.6917549   0.74775696]
      [-0.9622242   0.67038894]]]
    states = (3, 2)
    [[-0.9448845  -0.6292621 ]
     [-0.7216307   0.77867275]
     [-0.9622242   0.67038894]]
    

### 3. EX - many to one


```python
# data
xa = [[1],[2],[3]]; ya = [4]
xb = [[2],[3],[4]]; yb = [5]
xc = [[3],[4],[5]]; yc = [6]

x = np.array([xa, xb, xc])
y = np.array([ya, yb, yc])

print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")

# make a model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,1), name='input'),
    tf.keras.layers.SimpleRNN(100, name='RNN'),
    tf.keras.layers.Dense(1, name='output')
])
model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# run model
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    x.shape = (3, 3, 1)
    y.shape = (3, 1)
    WARNING:tensorflow:Please add `keras.layers.InputLayer` instead of `keras.Input` to Sequential model. `keras.Input` is intended to be used by Functional model.
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    RNN (SimpleRNN)              (None, 100)               10200     
    _________________________________________________________________
    output (Dense)               (None, 1)                 101       
    =================================================================
    Total params: 10,301
    Trainable params: 10,301
    Non-trainable params: 0
    _________________________________________________________________
    1, 2, 3 next? [[3.9146965]]
    2, 3, 4 next? [[5.1570582]]
    3, 4, 5 next? [[5.885913]]
    

- Functional API 버전


```python
# make a model
input_layer = tf.keras.Input(shape=(3,1), name='input')
rnn_layer = tf.keras.layers.SimpleRNN(100, name='RNN')(input_layer)
output_layer = tf.keras.layers.Dense(1, name='output')(rnn_layer)
model = tf.keras.Model(input_layer, output_layer)
model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# run model
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           [(None, 3, 1)]            0         
    _________________________________________________________________
    RNN (SimpleRNN)              (None, 100)               10200     
    _________________________________________________________________
    output (Dense)               (None, 1)                 101       
    =================================================================
    Total params: 10,301
    Trainable params: 10,301
    Non-trainable params: 0
    _________________________________________________________________
    1, 2, 3 next? [[3.9363065]]
    2, 3, 4 next? [[5.199828]]
    3, 4, 5 next? [[5.904044]]
    

- Stacked many to one


```python
# make a model
input_layer = tf.keras.Input(shape=(3,1), name='input')
rnn_layer0 = tf.keras.layers.SimpleRNN(100, return_sequences=True, name='RNN0')(input_layer)
rnn_layer1 = tf.keras.layers.SimpleRNN(100, name='RNN1')(rnn_layer0)  # layer 하나 추가
output_layer = tf.keras.layers.Dense(1, name='output')(rnn_layer1)
model = tf.keras.Model(input_layer, output_layer)
model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# run model
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           [(None, 3, 1)]            0         
    _________________________________________________________________
    RNN0 (SimpleRNN)             (None, 3, 100)            10200     
    _________________________________________________________________
    RNN1 (SimpleRNN)             (None, 100)               20100     
    _________________________________________________________________
    output (Dense)               (None, 1)                 101       
    =================================================================
    Total params: 30,401
    Trainable params: 30,401
    Non-trainable params: 0
    _________________________________________________________________
    1, 2, 3 next? [[3.9847925]]
    2, 3, 4 next? [[5.224788]]
    3, 4, 5 next? [[5.973739]]
    

### 4. EX - many to many


```python
# data
xa = [[1],[2],[3]]; ya = [[2],[3],[4]]
xb = [[2],[3],[4]]; yb = [[3],[4],[5]]
xc = [[3],[4],[5]]; yc = [[4],[5],[6]]

x = np.array([xa, xb, xc])
y = np.array([ya, yb, yc])

print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")

# make a model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,1), name='input'),
    tf.keras.layers.SimpleRNN(100, return_sequences=True, name='RNN'),
    tf.keras.layers.Dense(1, name='output')
])
model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# run model
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    x.shape = (3, 3, 1)
    y.shape = (3, 3, 1)
    WARNING:tensorflow:Please add `keras.layers.InputLayer` instead of `keras.Input` to Sequential model. `keras.Input` is intended to be used by Functional model.
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    RNN (SimpleRNN)              (None, 3, 100)            10200     
    _________________________________________________________________
    output (Dense)               (None, 3, 1)              101       
    =================================================================
    Total params: 10,301
    Trainable params: 10,301
    Non-trainable params: 0
    _________________________________________________________________
    1, 2, 3 next? [[[2.0947561]
      [2.9520147]
      [3.9683223]]]
    2, 3, 4 next? [[[3.0922108]
      [4.092709 ]
      [5.0780826]]]
    3, 4, 5 next? [[[3.8710687]
      [4.939366 ]
      [5.9261203]]]
    

- Functional API 버전


```python
# make a model
input_layer = tf.keras.Input(shape=(3,1), name='input')
rnn_layer = tf.keras.layers.SimpleRNN(100, return_sequences=True, name='RNN')(input_layer)

# 모든 time slice에 Dense layer를 적용
# 예측값이 여러개이므로 TimeDistributed 사용
output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name='output')(rnn_layer)
model = tf.keras.Model(input_layer, output_layer)
model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# run model
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    Model: "model_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           [(None, 3, 1)]            0         
    _________________________________________________________________
    RNN (SimpleRNN)              (None, 3, 100)            10200     
    _________________________________________________________________
    output (TimeDistributed)     (None, 3, 1)              101       
    =================================================================
    Total params: 10,301
    Trainable params: 10,301
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:5 out of the last 13 calls to <function Model.make_predict_function.<locals>.predict_function at 0x000001DCED4313A0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    1, 2, 3 next? [[[1.9926912]
      [2.965968 ]
      [3.9874735]]]
    2, 3, 4 next? [[[3.0715206]
      [4.1023574]
      [5.0892925]]]
    3, 4, 5 next? [[[3.9580004]
      [4.9611974]
      [5.985474 ]]]
    

- Stacked many to many


```python
# make a model
input_layer = tf.keras.Input(shape=(3,1), name='input')
rnn_layer0 = tf.keras.layers.SimpleRNN(100, return_sequences=True, name='RNN0')(input_layer)
rnn_layer1 = tf.keras.layers.SimpleRNN(100, return_sequences=True, name='RNN1')(rnn_layer0)
output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name='output')(rnn_layer1)
model = tf.keras.Model(input_layer, output_layer)
model.summary()

# compile
model.compile(loss='mse', optimizer='adam')

# run model
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    Model: "model_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           [(None, 3, 1)]            0         
    _________________________________________________________________
    RNN0 (SimpleRNN)             (None, 3, 100)            10200     
    _________________________________________________________________
    RNN1 (SimpleRNN)             (None, 3, 100)            20100     
    _________________________________________________________________
    output (TimeDistributed)     (None, 3, 1)              101       
    =================================================================
    Total params: 30,401
    Trainable params: 30,401
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:5 out of the last 13 calls to <function Model.make_predict_function.<locals>.predict_function at 0x000001DCEC2BADC0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    1, 2, 3 next? [[[1.8675102]
      [2.9398968]
      [3.9066122]]]
    2, 3, 4 next? [[[3.0823917]
      [4.0590305]
      [5.0320225]]]
    3, 4, 5 next? [[[3.982791 ]
      [4.903591 ]
      [5.8915787]]]
    

## D. LSTM

#### *Vanishing gradient problem*
- RNN은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 역전파시 gradient가 점차 줄어 학습능력이 크게 저하
- 즉, RNN은 long-term dependency가 없음

### LSTM
- Vanishing gradient problem를 극복하기 위한 것. 즉, 오래전의 정보도 기억할 수 있는 long-term dependency가 있음
- RNN의 **hidden state에 cell-state(기억)** 와 3개의 gate를 추가한 구조
- cell state가 hidden state 앞에서 처리
- 기본 아이디어 : 잊을 것은 잊고 중요한 정보만 기억하는 것이 효율적임
- cell state의 값을 제어하는 3개의 gate들\
  − input gate : 새로운 정보 중에서 어떤 것을 cell state에 담을 것인지 결정하는 게이트\
  − forget gate : 잊어버리기로 한 것을 가중치를 사용하여 실제로 잊어버리는 게이트\
  − output gate : 어떤 값을 출력할지 결정하는 게이트

> keras를 이용한 LSTM 구현 (SimpleRNN 대신 LSTM을 사용)


```python
# data
xa = [[1],[2],[3]]; ya = [4]
xb = [[2],[3],[4]]; yb = [5]
xc = [[3],[4],[5]]; yc = [6]

x = np.array([xa, xb, xc])
y = np.array([ya, yb, yc])

# make a model
input_layer = tf.keras.Input(shape=(3,1), name='input')
rnn_layer = tf.keras.layers.LSTM(100, name='RNN')(input_layer)  # LSTM
output_layer = tf.keras.layers.Dense(1, name='output')(rnn_layer)
model = tf.keras.Model(input_layer, output_layer)
model.summary()

# compile & run
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=0)

# test model
print(f"1, 2, 3 next? {model.predict([[[1],[2],[3]]])}")
print(f"2, 3, 4 next? {model.predict([[[2],[3],[4]]])}")
print(f"3, 4, 5 next? {model.predict([[[3],[4],[5]]])}")
```

    Model: "model_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           [(None, 3, 1)]            0         
    _________________________________________________________________
    RNN (LSTM)                   (None, 100)               40800     
    _________________________________________________________________
    output (Dense)               (None, 1)                 101       
    =================================================================
    Total params: 40,901
    Trainable params: 40,901
    Non-trainable params: 0
    _________________________________________________________________
    1, 2, 3 next? [[3.88253]]
    2, 3, 4 next? [[5.128863]]
    3, 4, 5 next? [[5.9369135]]
    


```python

```
