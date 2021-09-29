---

layout: posts
title:  "데이터 분석 심화 2 - 신경망의 이해"
categories: ['algorithm']
tags: [mlp, linear regression, multiple linear regression, logistic regression]
---




## A. Perceptron

### 1. 퍼셉트론(Perceptron): 인공 뉴런(Artificial Neuron)
- 여러 개의 입력된 정보에 대해 가중치 합을 계산하여 출력정보 생성
- 퍼셉트론의 출력: 활성화 함수(Activation Function) 혹은 전달함수 (Transfer Function)을 통해 전달

### 2. 퍼셉트론(Perceptron)의 동작원리

![화면 캡처 2021-07-14 093717](https://user-images.githubusercontent.com/86539195/125592413-eaa10854-f806-4cf8-91b8-2fe64f21b83d.png)

- 그림의 원 : 뉴런(neuron) / 노드(node)
- 가중치는 전류의 저항과 비슷한 의미(가중치가 크면 더 강한신호를 흘려 보냄)
- 뉴런에서 보내온 신호의 총합이 θ(임계값)보다 큰 경우 ⇒ 1을 출력(활성화)

### 3. Perceptron 구현
- AND (0.5, 0.5, -0.7)


```python
import numpy as np
```


```python
def AND(x1, x2):
    x = np.array([x1, x2]) 
    w = np.array([0.5, 0.5]) 
    b = -0.7 
    y = np.sum(w*x) + b
    if y > 0:
        return 1
    else:
        return 0
    
def main():
    print("(0, 0) => ", AND(0,0))
    print("(0, 1) => ", AND(0,1))
    print("(1, 0) => ", AND(1,0))
    print("(1, 1) => ", AND(1,1))
    
main()
```

    (0, 0) =>  0
    (0, 1) =>  0
    (1, 0) =>  0
    (1, 1) =>  1
    

## B. MLP(Multiple Layer Perceptron)

### 1. 신경망(Neural-Net)
- 신경망의 구성 입력층 – 은닉층 – 출력층
- 가중치를 갖는 층은 입력층과 은닉층뿐이므로 2층 신경망이라고도 함

### 2. 간단한 신경망 구현하기
- 행렬의 곱셈으로 신경망 계산을 수행


```python
X = np.array([1,2])
W = np.array([[1,2,3],[4,5,6]])
Y = np.dot(X, W)

# shape : tuple에 요소가 1개이면 콤마를 뒤에 붙임
print('shape of X =', X.shape)
print(X)
print('shape of W =', W.shape)
print(W)
print('shape of Y =', Y.shape)
print(Y)
```

    shape of X = (2,)
    [1 2]
    shape of W = (2, 3)
    [[1 2 3]
     [4 5 6]]
    shape of Y = (3,)
    [ 9 12 15]
    

- 퍼셉트론에 편향(bias) 표현하기

![화면 캡처 2021-07-14 095843](https://user-images.githubusercontent.com/86539195/125592452-6090e713-cab9-431d-8dc7-f747d94b61bf.png)

### 3. 활성함수의 종류
- Sigmoid function
- ReLU function (Rectified Linear Unit)
- 쌍곡탄젠트 function: h(z) = tanh(z)

### 4. 다층 퍼셉트론의 구현

#### step1: MLP구현(1번 레이어)


```python
# 활성함수 적용 - sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# input nodes
X = np.array([1.0, 0.5])

# compute layer1 nodes
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1

# 활성함수 적용
Z1 = sigmoid(A1)

print(A1)
print(Z1)
```

    [0.3 0.7 1.1]
    [0.57444252 0.66818777 0.75026011]
    


```python
# 참고) 활성함수 적용 - relu function
def relu(a):
    if x>0:
        return x
    else:
        return 0
```

#### step2: MLP구현(2번 레이어)
- 1단계의 출력(Z1)을 2단계의 입력으로 사용


```python
# compute Layer2 nodes
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2

Z2 = sigmoid(A2)

print(A2)
print(Z2)
```

    [0.51615984 1.21402696]
    [0.62624937 0.7710107 ]
    

#### step3: MLP구현(출력 레이어)


```python
# compute output layer nodes
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3

print(A3)
```

    [0.31682708 0.69627909]
    

#### ※ 참고: softmax function
- 출력층의 값을 어떻게 출력할지를 결정
- 출력값들을 확률로 변환
- 주로 분류(classification) 문제에 사용됨


```python
def softmax(a):
    m = np.max(a)
    exp_a = np.exp(a - m) # overflow 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

print(softmax(A3))
```

    [0.40625907 0.59374093]
    

## C. Linear Regression

### 1. Regression이란?

- 회귀분석(Regression Analysis) \
  : 2개 또는 그 이상 변수들의 의존관계를 파악함으로써 특정 변수(종속변수)의 값을 예측하는 통계학의 한 분야

- Linear Regression Analysis(선형 회귀분석):\
  : 두 변수 x, y에 대한 n개의 측정값 (x1, y1), (x2, y2), ···, (xn, yn)이 있을 때 주어진 가설(hypothesis)에 대한 비용(cost)이 최소화 되도록 하는 직선을 찾는 문제

- Linear Regression이란? \
  cost함수 cost W, b 를 최소화하는 W와 b를 찾는 문제

### 2. cost함수의 모양


```python
import tensorflow as tf
import matplotlib.pyplot as plt
```


```python
X = tf.constant([1,2,3,4,5], dtype=tf.float32)
Y = tf.constant([1,2,3,4,5], dtype=tf.float32)

def run(W):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    return cost.numpy()

W_val = []
cost_val = []
for i in range(-30, 50):
    w = i * 0.1
    c = run(w)
    
    if i%10 == 0:
        print('w = ', w, end='\t')
        print('c = ', c)
    W_val.append(w)
    cost_val.append(c)
    
plt.plot(W_val, cost_val)
plt.show()
```

    w =  -3.0	c =  176.0
    w =  -2.0	c =  99.0
    w =  -1.0	c =  44.0
    w =  0.0	c =  11.0
    w =  1.0	c =  0.0
    w =  2.0	c =  11.0
    w =  3.0	c =  44.0
    w =  4.0	c =  99.0
    


    
![output_26_1](https://user-images.githubusercontent.com/86539195/125592515-0529e9d6-bb9c-4e63-987c-96c3d64b292e.png)
    


- 위 예제에서는 W값을 for문을 이용해 cost함수가 최소가 되는 점을 사용
- 경사하강알고리즘을 이용 (Gradient descent algorithm)\
  : 임의의 곳에서 시작하여 경사도(gradient)에 따라 W를 변경시켜가면서 cost 함수의 값이 최소화되는 W를 구하는 알고리즘

### 3. Linear Regression(Tensorflow로 구현)
#### ★★ Flow 중요 ★★
1. Graph 구성(가설, 비용함수, 학습함수 정의)
 - Hypothesis(가설) 정의: 여기에 사용되는 변수(Variable)생성
 - Cost/Loss함수 정의
 - Train함수 정의
2. Training
 - 충분한 만큼 반복하면서 Graph상의 Tensor들을 실행, Train 함수 실행, Loss 함수 실행
 - 결과를 출력(또는, Tensorboard로 확인)
3. Testing


#### (1) Graph 구성
- Hypothesis(가설) 정의


```python
W = tf.Variable(tf.random.normal([1]), name='Weight')
B = tf.Variable(tf.random.normal([1]), name='Bias')

print(W.numpy(), B.numpy())
```

    [-1.2062066] [1.2362899]
    


```python
# tf.function
def Hypothesis(X):
    return W * X + B
```

- Cost/Loss함수 정의


```python
# tf.function
def loss(H, Y):
    return tf.reduce_mean(tf.square(H - Y))
```

- Train함수 정의


```python
def train(X, Y, learning_rate=0.01):
    with tf.GradientTape() as tape:
        _loss = loss(Hypothesis(X), Y)
    _w, _b = tape.gradient(_loss, [W, B])
    W.assign_sub(learning_rate * _w) # 텐서의 값 변경을 위한 메소드
    B.assign_sub(learning_rate * _b) # assign(=), assign_add(+=),assign_sub(-=)
```

#### (2) Training
- Graph상의 Tensor들을 실행: Train 함수 실행, Loss 함수 실행
- 결과를 출력(또는, Tensorboard로 확인)


```python
for step in range(2001):
    train(X, Y, learning_rate=0.01)
    _c = loss(Hypothesis(X), Y)
    if step % 100 == 0:
              print(f"{step}: {_c.numpy()} {W.numpy()} {B.numpy()}")
print('\nfinal W =', W.numpy(), 'b =', B.numpy())
```

    0: 22.774829864501953 [-0.79501855] [1.3439364]
    100: 0.2703404128551483 [0.66243804] [1.2187058]
    200: 0.13732418417930603 [0.75941336] [0.86859417]
    300: 0.06975621730089188 [0.8285295] [0.61906296]
    400: 0.03543388098478317 [0.87778986] [0.44121754]
    500: 0.01799924671649933 [0.91289854] [0.31446388]
    600: 0.009143054485321045 [0.93792117] [0.22412415]
    700: 0.004644359461963177 [0.95575535] [0.15973736]
    800: 0.002359188161790371 [0.96846604] [0.11384772]
    900: 0.0011983870062977076 [0.9775252] [0.08114132]
    1000: 0.0006087412475608289 [0.9839818] [0.05783092]
    1100: 0.0003092212718911469 [0.9885835] [0.04121713]
    1200: 0.00015707581769675016 [0.99186325] [0.02937626]
    1300: 7.978892972460017e-05 [0.99420077] [0.02093699]
    1400: 4.053045267937705e-05 [0.9958668] [0.01492223]
    1500: 2.058845348074101e-05 [0.9970542] [0.01063533]
    1600: 1.045843782776501e-05 [0.9979005] [0.00758]
    1700: 5.312349912856007e-06 [0.9985036] [0.00540236]
    1800: 2.6983966563420836e-06 [0.9989335] [0.00385035]
    1900: 1.3708852293348173e-06 [0.99923986] [0.00274428]
    2000: 6.963424539208063e-07 [0.99945825] [0.00195596]
    
    final W = [0.99945825] b = [0.00195596]
    

#### (3) Testing


```python
test_data = [2, 4, 1, 5, 3]
for data in test_data:
    y = data * W.numpy() + B.numpy()
    print("X =", data, "then Y =", y)
```

    X = 2 then Y = [2.0008724]
    X = 4 then Y = [3.999789]
    X = 1 then Y = [1.0014142]
    X = 5 then Y = [4.999247]
    X = 3 then Y = [3.0003307]
    

## D. Multiple Linear Regression
- 입력데이터가 1개가 아니라 여러 개인 경우

### (1) Graph 구성
- Hypothesis(가설) 정의


```python
x_data = tf.constant([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]], dtype=tf.float32)

y_data = tf.constant([[152],[185],[180],[196],[142]], dtype=tf.float32)

W = tf.Variable(tf.random.normal([3, 1]), name='weight')
B = tf.Variable(tf.random.normal([1]), name='bias')

# tf.function
def Hypothesis(X):
    return tf.matmul(X, W) + B
```

- Cost/Loss함수 정의


```python
def loss(H, Y):
    return tf.reduce_mean(tf.square(H - Y))
```

- Train함수 정의


```python
def train(X, Y, learning_rate=1e-5):
    with tf.GradientTape() as tape:
        _loss = loss(Hypothesis(X), Y)
    _w, _b = tape.gradient(_loss, [W, B])
    W.assign_sub(learning_rate * _w)
    B.assign_sub(learning_rate * _b)
```

### (2) Training


```python
for step in range(10001):
    # _h = Hypothesis(x_data)
    _c = loss(Hypothesis(x_data), y_data)
    train(x_data, y_data, learning_rate=1e-5)
    if step % 500 == 0:
        print(f"{step}: {_c.numpy()}")
```

    0: 67437.3359375
    500: 17.765220642089844
    1000: 13.725598335266113
    1500: 10.641489028930664
    2000: 8.286285400390625
    2500: 6.486988067626953
    3000: 5.111687183380127
    3500: 4.0598320960998535
    4000: 3.2546939849853516
    4500: 2.637813091278076
    5000: 2.16453218460083
    5500: 1.8008434772491455
    6000: 1.520800232887268
    6500: 1.304614543914795
    7000: 1.1371989250183105
    7500: 1.007016658782959
    8000: 0.9053038358688354
    8500: 0.825353741645813
    9000: 0.7620664834976196
    9500: 0.7115278244018555
    10000: 0.6707784533500671
    

### (3) Testing


```python
test_data = tf.constant(
[[ 73, 80, 75],
[ 93, 88, 93],
[ 89, 91, 90],
[ 96, 98, 100],
[ 73, 66, 70],
[100, 70, 101],
[ 60, 70, 110],
[ 90, 100, 80]], dtype=tf.float32)

print("Testing...")
for data in test_data:
    y = Hypothesis([data])
    print("X =", data.numpy(), "then Y =", y.numpy())
```

    Testing...
    X = [73. 80. 75.] then Y = [[152.56903]]
    X = [93. 88. 93.] then Y = [[184.04326]]
    X = [89. 91. 90.] then Y = [[181.14336]]
    X = [ 96.  98. 100.] then Y = [[195.16493]]
    X = [73. 66. 70.] then Y = [[142.33139]]
    X = [100.  70. 101.] then Y = [[179.57645]]
    X = [ 60.  70. 110.] then Y = [[127.155136]]
    X = [ 90. 100.  80.] then Y = [[189.42307]]
    

## E. Logistic Regression

### 1. Binary Logistic Classification
- Class가 2개인 경우: 주어진 데이터가 어느 클래스에 속하는가?, Class(label)는 보통 0과 1로 encoding
- 새로운 Hypothesis: H x = Wx + b, 이 가설은 0보다 작거나 1보다 큰 경우가 있음, 즉, 0 ~ 1사이의 값으로 만드는 함수가 필요

### 2. Logisitic Regression(Tensorflow로구현)

1. Graph 구성(가설, 비용함수, 학습함수 정의)
 - Hypothesis(가설) 정의: 여기에 사용되는 변수(Variable)생성
 - Cost/Loss함수 정의
 - Train함수 정의
2. Training
 - 충분한 만큼 반복하면서 Graph상의 Tensor들을 실행, Train 함수 실행, Loss 함수 실행
 - 결과를 출력(또는, Tensorboard로 확인)
3. Testing


#### (1) Graph 구성
- Hypothesis(가설) 정의


```python
# training data
x_data = tf.constant([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], dtype=tf.float32)
y_data = tf.constant([[0], [0], [0], [1], [1], [1]], dtype=tf.float32)

W = tf.Variable(tf.random.normal([2, 1]), name='weight')
B = tf.Variable(tf.random.normal([1]), name='bias')

# define hypothesis
# tf.function
def Hypothesis(X):
    return tf.sigmoid(tf.matmul(X, W) + B)
```

- Cost/Loss함수 정의


```python
# define cost function
# tf.function
def loss(H, Y):
    cost = -tf.reduce_mean(Y * tf.math.log(H) + (1 - Y) * tf.math.log(1 - H))
    return cost
```

- cost를 최소화시키도록 학습하는 함수 정의


```python
# minimize the cost function
# tf.function
def train(X, Y, learning_rate=0.1):
    with tf.GradientTape() as tape:
        _loss = loss(Hypothesis(X), Y)
    _w, _b = tape.gradient(_loss, [W, B])
    W.assign_sub(learning_rate * _w)
    B.assign_sub(learning_rate * _b)
```

#### (2) Training


```python
# training...
for step in range(10001):
    _c = loss(Hypothesis(x_data), y_data)
    train(x_data, y_data, learning_rate=0.1)
    if step % 500 == 0:
        print(f"{step}: {_c.numpy()}")
```

    0: 2.555386781692505
    500: 0.22269050776958466
    1000: 0.14290624856948853
    1500: 0.10526429861783981
    2000: 0.08352311700582504
    2500: 0.06935404241085052
    3000: 0.05937018617987633
    3500: 0.05194373428821564
    4000: 0.04619646072387695
    4500: 0.041612133383750916
    5000: 0.03786744549870491
    5500: 0.03474919870495796
    6000: 0.03211116045713425
    6500: 0.0298495814204216
    7000: 0.027888618409633636
    7500: 0.02617173083126545
    8000: 0.0246556606143713
    8500: 0.02330699749290943
    9000: 0.022099250927567482
    9500: 0.021011322736740112
    10000: 0.02002612315118313
    

#### (3) Testing


```python
# accuracy computation
# tf.function
def test(H, Y):
    # True if H > 0.5 else False
    predicted = tf.cast(H > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y),dtype=tf.float32))
    return predicted, accuracy

# report accuracy...
print("\nAccuracy...")

_h = Hypothesis(x_data)
_p, _a = test(_h, y_data)

print("Hypothesis =\n", _h.numpy())
print("Predicted =\n", _p.numpy())
print("\nAccuracy =", _a.numpy())
```

    
    Accuracy...
    Hypothesis =
     [[2.8303266e-04]
     [3.1894714e-02]
     [3.9355338e-02]
     [9.5582885e-01]
     [9.9820006e-01]
     [9.9968076e-01]]
    Predicted =
     [[0.]
     [0.]
     [0.]
     [1.]
     [1.]
     [1.]]
    
    Accuracy = 1.0
    


```python

```
