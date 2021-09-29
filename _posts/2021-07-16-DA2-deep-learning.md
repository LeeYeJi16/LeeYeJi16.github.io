---
layout: posts
title:  "데이터 분석 심화 2 - 딥러닝 프로그램"
categories: ['algorithm']
tags: [deep learning]
---

# 데이터 분석 심화 2 - 딥러닝 프로그램

## A. 딥러닝을 이용한 분류 

### 1. 딥러닝 모델 적용 단계
- **단계 1: 데이터 로딩**
- **단계 2: 데이터 전처리**
- **단계 3: 모델 정의(define)**\
  Create a sequential model\
  Add layers, each layer (one or more convolution, pooling, batch normalization, activation function)
- **단계 4: 모델 컴파일(compile)**\
  Apply the loss function and optimizer before calling compile\
  *분류모델 > Adam, 회귀모델 > RMSProp*

- **단계 5: 모델 적합(fit)**\
  Fit the model with training data
- **단계 6: 모델 평가(evaluate)**\
  evaluate()
- **단계 7: 예측 하기**\
  predict()
- **단계 8: 모델 저장**

### 2. 딥러닝을 위한 분류 (cifar10)

- 데이터 로딩


```python
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
```


```python
#Label	Description
#0	airplane
#1	automobile
#2	bird
#3	cat
#4	deer
#5	dog
#6	frog
#7	horse
#8	ship
#9	truck
```


```python
np.random.seed(100)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)
```

    (50000, 32, 32, 3)
    (50000, 1)
    (10000, 32, 32, 3)
    (10000, 1)
    


```python
# (50000, 32, 32, 3) 
# 32픽셀 그림
# 3 >>  R, G, B

for i in range(5):
    print("y_train : ", y_train[i])
    print("x_train : ", x_train[i])
```

    y_train :  [6]
    x_train :  [[[ 59  62  63]
      [ 43  46  45]
      [ 50  48  43]
      ...
      [158 132 108]
      [152 125 102]
      [148 124 103]]
    
     [[ 16  20  20]
      [  0   0   0]
      [ 18   8   0]
      ...
      [123  88  55]
      [119  83  50]
      [122  87  57]]
    
     [[ 25  24  21]
      [ 16   7   0]
      [ 49  27   8]
      ...
      [118  84  50]
      [120  84  50]
      [109  73  42]]
    
     ...
    
     [[208 170  96]
      [201 153  34]
      [198 161  26]
      ...
      [160 133  70]
      [ 56  31   7]
      [ 53  34  20]]
    
     [[180 139  96]
      [173 123  42]
      [186 144  30]
      ...
      [184 148  94]
      [ 97  62  34]
      [ 83  53  34]]
    
     [[177 144 116]
      [168 129  94]
      [179 142  87]
      ...
      [216 184 140]
      [151 118  84]
      [123  92  72]]]
    y_train :  [9]
    x_train :  [[[154 177 187]
      [126 137 136]
      [105 104  95]
      ...
      [ 91  95  71]
      [ 87  90  71]
      [ 79  81  70]]
    
     [[140 160 169]
      [145 153 154]
      [125 125 118]
      ...
      [ 96  99  78]
      [ 77  80  62]
      [ 71  73  61]]
    
     [[140 155 164]
      [139 146 149]
      [115 115 112]
      ...
      [ 79  82  64]
      [ 68  70  55]
      [ 67  69  55]]
    
     ...
    
     [[175 167 166]
      [156 154 160]
      [154 160 170]
      ...
      [ 42  34  36]
      [ 61  53  57]
      [ 93  83  91]]
    
     [[165 154 128]
      [156 152 130]
      [159 161 142]
      ...
      [103  93  96]
      [123 114 120]
      [131 121 131]]
    
     [[163 148 120]
      [158 148 122]
      [163 156 133]
      ...
      [143 133 139]
      [143 134 142]
      [143 133 144]]]
    y_train :  [9]
    x_train :  [[[255 255 255]
      [253 253 253]
      [253 253 253]
      ...
      [253 253 253]
      [253 253 253]
      [253 253 253]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [254 254 254]
      [254 254 254]
      ...
      [254 254 254]
      [254 254 254]
      [254 254 254]]
    
     ...
    
     [[113 120 112]
      [111 118 111]
      [105 112 106]
      ...
      [ 72  81  80]
      [ 72  80  79]
      [ 72  80  79]]
    
     [[111 118 110]
      [104 111 104]
      [ 99 106  98]
      ...
      [ 68  75  73]
      [ 70  76  75]
      [ 78  84  82]]
    
     [[106 113 105]
      [ 99 106  98]
      [ 95 102  94]
      ...
      [ 78  85  83]
      [ 79  85  83]
      [ 80  86  84]]]
    y_train :  [4]
    x_train :  [[[ 28  25  10]
      [ 37  34  19]
      [ 38  35  20]
      ...
      [ 76  67  39]
      [ 81  72  43]
      [ 85  76  47]]
    
     [[ 33  28  13]
      [ 34  30  14]
      [ 32  27  12]
      ...
      [ 95  82  55]
      [ 96  82  56]
      [ 85  72  45]]
    
     [[ 39  32  15]
      [ 40  33  17]
      [ 57  50  33]
      ...
      [ 93  76  52]
      [107  89  66]
      [ 95  77  54]]
    
     ...
    
     [[ 83  73  52]
      [ 87  77  56]
      [ 84  74  52]
      ...
      [ 99  93  70]
      [ 90  84  61]
      [ 81  75  52]]
    
     [[ 88  72  51]
      [ 90  74  52]
      [ 93  77  56]
      ...
      [ 80  74  53]
      [ 76  70  49]
      [ 82  76  55]]
    
     [[ 97  78  56]
      [ 94  75  53]
      [ 93  75  53]
      ...
      [ 54  47  28]
      [ 63  56  37]
      [ 72  65  46]]]
    y_train :  [1]
    x_train :  [[[170 180 198]
      [168 178 196]
      [177 185 203]
      ...
      [162 179 215]
      [158 178 214]
      [157 177 212]]
    
     [[168 181 198]
      [172 185 201]
      [171 183 200]
      ...
      [159 177 212]
      [156 176 211]
      [154 174 209]]
    
     [[154 170 186]
      [149 165 181]
      [129 144 162]
      ...
      [161 178 214]
      [157 177 212]
      [154 174 209]]
    
     ...
    
     [[ 74  84  80]
      [ 76  85  81]
      [ 78  85  82]
      ...
      [ 71  75  78]
      [ 68  72  75]
      [ 61  65  68]]
    
     [[ 68  76  77]
      [ 69  77  78]
      [ 72  79  78]
      ...
      [ 76  80  83]
      [ 71  75  78]
      [ 71  75  78]]
    
     [[ 67  75  78]
      [ 68  76  79]
      [ 69  75  76]
      ...
      [ 75  79  82]
      [ 71  75  78]
      [ 73  77  80]]]
    


```python
import matplotlib.pyplot as plt 
%matplotlib inline 

for i in range(5): 
    plt.imshow(x_train[i]) 
    print(y_train[i], end=", ") 
    plt.show()
```

    [6], 


    
![output_8_1](https://user-images.githubusercontent.com/86539195/135232982-0a50638b-7dc4-446d-88cd-443f5662af1b.png)
    


    [9], 


    
![output_8_3](https://user-images.githubusercontent.com/86539195/135233001-21fbbecf-fe73-4e65-bcc4-981734545765.png)
    


    [9], 


    
![output_8_5](https://user-images.githubusercontent.com/86539195/135233018-327756d5-3881-4058-b737-61a8c384543c.png)
    


    [4], 


    
![output_8_7](https://user-images.githubusercontent.com/86539195/135233045-1c02c471-ee80-4766-af7b-7ed567462b28.png)
    


    [1], 


    
![output_8_9](https://user-images.githubusercontent.com/86539195/135233073-54b4d27f-3acd-4e4e-9e8a-1707e79fab70.png)
    


- 데이터 전처리


```python
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
```


```python
# Gaussian Normalization (Z-score)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
```


```python
# class vector를 binary class matrics로 변환 (one-hot vector로 변환)
labels = 10
y_train = to_categorical(y_train, labels)
y_test = to_categorical(y_test, labels)

print("y_train : ", y_train)
print("y_test : ", y_test)
```

    y_train :  [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     ...
     [0. 0. 0. ... 0. 0. 1.]
     [0. 1. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]]
    y_test :  [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 1. 0.]
     [0. 0. 0. ... 0. 1. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 1. 0. 0.]]
    

- 모델 정의


```python
# 혹시 이미 그려둔 그래프가 있다면 날려줘!
tf.keras.backend.clear_session()

# model에 순차적으로 레이어를 쌓아가겠다는 의도!
model = Sequential()

# Sequential()을 사용하면, Input Layer는 자동으로 설계한다고 생각하면 됨!

# 첫번째 Hidden 레이어를 add할 때는 input의 shape를 항상 정해줘야 한다!
### 노드 수를 마음 껏 조절해보자!
model.add(Dense(512, input_shape=(3072,), activation='relu', name = 'Hidden1'))
# name = 'Hidden1' 레이어에 이름을 붙일 수 있다. 나중에 summary할 때 깔끔하다.

### 노드 수를 마음 껏 조절해보자!
model.add(Dense(120, activation = 'relu', name = 'Hidden2'))
model.add( Dense(512, activation = 'relu', name = 'Hidden3') )

# output Layer
model.add(Dense(10, activation = 'sigmoid'))
```

- 모델 컴파일


```python
# model comfile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics =['accuracy'])

print('히든 레이어를 여러개 갖는 딥러닝 모델')
print('딥러닝이라 불리는 모델들 중 가장 기본 구조이다.')
model.summary()
```

    히든 레이어를 여러개 갖는 딥러닝 모델
    딥러닝이라 불리는 모델들 중 가장 기본 구조이다.
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    Hidden1 (Dense)              (None, 512)               1573376   
    _________________________________________________________________
    Hidden2 (Dense)              (None, 120)               61560     
    _________________________________________________________________
    Hidden3 (Dense)              (None, 512)               61952     
    _________________________________________________________________
    dense (Dense)                (None, 10)                5130      
    =================================================================
    Total params: 1,702,018
    Trainable params: 1,702,018
    Non-trainable params: 0
    _________________________________________________________________
    

- 모델 적합


```python
# model fit
model.fit(x_train, y_train, batch_size = 1024, epochs=10, validation_data=(x_test, y_test), verbose = 1)

# verbose=0, 진행상황을 출력하지 않음
# verbose=2, 진행상황 결과만 출력함
```

    Epoch 1/10
    49/49 [==============================] - 37s 689ms/step - loss: 1.7397 - accuracy: 0.3847 - val_loss: 1.5351 - val_accuracy: 0.4558
    Epoch 2/10
    49/49 [==============================] - 19s 382ms/step - loss: 1.4659 - accuracy: 0.4844 - val_loss: 1.4631 - val_accuracy: 0.4864
    Epoch 3/10
    49/49 [==============================] - 33s 670ms/step - loss: 1.3427 - accuracy: 0.5280 - val_loss: 1.4219 - val_accuracy: 0.5051
    Epoch 4/10
    49/49 [==============================] - 33s 682ms/step - loss: 1.2425 - accuracy: 0.5645 - val_loss: 1.3922 - val_accuracy: 0.5110
    Epoch 5/10
    49/49 [==============================] - 26s 533ms/step - loss: 1.1484 - accuracy: 0.5978 - val_loss: 1.3785 - val_accuracy: 0.5196
    Epoch 6/10
    49/49 [==============================] - 22s 439ms/step - loss: 1.0658 - accuracy: 0.6283 - val_loss: 1.4056 - val_accuracy: 0.5164
    Epoch 7/10
    49/49 [==============================] - 19s 396ms/step - loss: 0.9906 - accuracy: 0.6527 - val_loss: 1.4092 - val_accuracy: 0.5206
    Epoch 8/10
    49/49 [==============================] - 18s 376ms/step - loss: 0.9180 - accuracy: 0.6808 - val_loss: 1.4212 - val_accuracy: 0.5274
    Epoch 9/10
    49/49 [==============================] - 18s 373ms/step - loss: 0.8442 - accuracy: 0.7074 - val_loss: 1.4742 - val_accuracy: 0.5157
    Epoch 10/10
    49/49 [==============================] - 18s 363ms/step - loss: 0.7749 - accuracy: 0.7295 - val_loss: 1.4818 - val_accuracy: 0.5295
    




    <tensorflow.python.keras.callbacks.History at 0x2da2480a250>



- 모델 평가


```python
# 모델 평가
score = model.evaluate(x_test, y_test, verbose = 1)
print('loss = ', score[0], 'test accuracy(정답율): ', score[1]*100)

# 결과는 설정해준 Hidden node 수에 따라 달라질 것
```

    313/313 [==============================] - 4s 14ms/step - loss: 1.4818 - accuracy: 0.5295 3s
    loss =  1.4817872047424316 test accuracy(정답율):  52.95000076293945
    

- 예측 하기


```python
# 결과 예측
result_predict = model.predict_classes(x_test)
print(result_predict)
```

    C:\Users\user\anaconda3\lib\site-packages\tensorflow\python\keras\engine\sequential.py:455: UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
      warnings.warn('`model.predict_classes()` is deprecated and '
    

    [3 8 0 ... 5 6 7]
    


```python
x_test = x_test.reshape(10000, 32, 32, 3)

for i in range(20): 
    plt.imshow(x_test[i]) 
    print(result_predict[i], end=", ") 
    plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    3, 


    
![output_23_2](https://user-images.githubusercontent.com/86539195/135233122-703d3f40-5cc1-4f8f-b697-b6f93f68c3c1.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    8, 


    
![output_23_5](https://user-images.githubusercontent.com/86539195/135233139-99642630-ca02-4ae0-9d01-be7d443946ed.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    0, 


    
![output_23_8](https://user-images.githubusercontent.com/86539195/135233161-7000dd9a-a2a9-4243-9b2a-f8fce10de895.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    8, 


    
![output_23_11](https://user-images.githubusercontent.com/86539195/135233179-ccb287cd-202b-457f-aea4-62fbaf02117e.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    4, 


    
![output_23_14](https://user-images.githubusercontent.com/86539195/135233201-4d289f73-e9b6-4ce2-b6a6-8b777859d0e9.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    6, 


    
![output_23_17](https://user-images.githubusercontent.com/86539195/135233216-878f26a3-18f4-49ae-8401-df5d9ed759af.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    1, 


    
![output_23_20](https://user-images.githubusercontent.com/86539195/135233228-b7a5b496-c87c-4e9d-9b47-1384f46b7189.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    6, 


    
![output_23_23](https://user-images.githubusercontent.com/86539195/135233242-8c95948e-2f04-4a73-a15b-dad2fc06056f.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    4, 


    
![output_23_26](https://user-images.githubusercontent.com/86539195/135233264-f0f7d7de-9157-4ad8-a96b-763892447794.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    1, 


    
![output_23_29](https://user-images.githubusercontent.com/86539195/135233276-53b35cd7-4ea0-4e22-87df-d2591e58bf78.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    0, 


    
![output_23_32](https://user-images.githubusercontent.com/86539195/135233302-abfb029c-e9d1-4c0d-ab4c-7492ac9d66c1.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    1, 


    
![output_23_35](https://user-images.githubusercontent.com/86539195/135233319-009d60dd-23f3-44cf-9ec2-397a794fb407.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    5, 


    
![output_23_38](https://user-images.githubusercontent.com/86539195/135233333-584e4e31-abcc-4219-99ef-a4764fd2703a.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    7, 


    
![output_23_41](https://user-images.githubusercontent.com/86539195/135233358-39471bcb-744b-475f-95d0-a8a114e1000c.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    9, 


    
![output_23_44](https://user-images.githubusercontent.com/86539195/135233379-c9ea5dda-4f30-440f-a318-5adb8c62f8da.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    8, 


    
![output_23_47](https://user-images.githubusercontent.com/86539195/135233400-1e5adaef-a35c-4bff-81fe-d7b925e9b667.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    7, 


    
![output_23_50](https://user-images.githubusercontent.com/86539195/135233416-4e04e0a4-82d8-4df9-bb8b-1a3b16796fda.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    3, 


    
![output_23_53](https://user-images.githubusercontent.com/86539195/135233435-a4855f1b-3d63-43da-84b7-700d9ae32b7f.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    8, 


    
![output_23_56](https://user-images.githubusercontent.com/86539195/135233448-b7f7d311-6c53-476b-999f-350eba26472a.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    

    6, 


    
![output_23_59](https://user-images.githubusercontent.com/86539195/135233467-fa68e621-f757-4170-ae9a-110d52617f33.png)
    



```python
#0	airplane
#1	automobile
#2	bird
#3	cat
#4	deer
#5	dog
#6	frog
#7	horse
#8	ship
#9	truck

# 위 결과에서 마지막 frog>6 으로 맞게 분류함
```

## B. Classification for MNIST

※ 참고 : epoch, batch_size\
데이터의 크기가 매우 크므로, 데이터를 몇개로 나누어서 학습시킴
- epoch: 전체 Dataset을 한 번 학습시키는 것을 1 epoch이라고 함(즉, 몇 번 반복해서 학습시킬지를 의미)
- batch size: 한 번에 메모리에서 처리하는 양
- number of iterations = number of training samples / batch_size\
  :(예) 전체 sample이 20,000개인 경우, 이것을 100개로 나누어 학습시키려면 batch_size는 200이 되어야 함


```python
import numpy as np
import os
import tensorflow as tf
```


```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```

    (60000, 28, 28)
    (60000,)
    (10000, 28, 28)
    (10000,)
    


```python
import matplotlib.pyplot as plt 
%matplotlib inline 

for i in range(5): 
    plt.imshow(x_train[i]) 
    print(y_train[i], end=", ") 
    plt.show()
```

    5, 


    
![output_29_1](https://user-images.githubusercontent.com/86539195/135233497-a1b549a0-224f-4e1a-b005-aecc4fec8f7a.png)
    


    0, 


    
![output_29_3](https://user-images.githubusercontent.com/86539195/135233509-5f657b9e-8ce3-4d46-afab-f76b286f12b9.png)
    


    4, 


    
![output_29_5](https://user-images.githubusercontent.com/86539195/135233520-37888f53-4b5a-4ca3-b61e-bec9e07f879a.png)
    


    1, 


    
![output_29_7](https://user-images.githubusercontent.com/86539195/135233533-c79d052c-e132-4ac7-abe3-4ebeab14f5fe.png)
    


    9, 


    
![output_29_9](https://user-images.githubusercontent.com/86539195/135233550-8947e669-4a85-46c6-b820-4d2ce3ba8aae.png)
    



```python
# 데이타 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

labels = 10

y_train = tf.one_hot(y_train, labels).numpy()
y_test = tf.one_hot(y_test, labels).numpy()
```


```python
# Hypothsis 정의

W = tf.Variable(tf.random.normal([784, labels]), name='weight')
B = tf.Variable(tf.random.normal([labels]), name='bias')

@tf.function
def Hypothesis(x):
    logits = tf.add(tf.matmul(tf.cast(x, tf.float32), W), B)
    return tf.nn.softmax(logits)
```


```python
# Cost 함수 정의

#tf.reduce_sum : 다차원의 요소들의 함계
#tf.reduce_mean : 다차원의 요소들의 평균

@tf.function
def loss(H, Y):
    entropy = -tf.reduce_sum(Y * tf.math.log(H), axis = 1)
    # entropy = tf.losses.categorical_crossentropy(Y, H)
    cost = tf.reduce_mean(entropy)
    return cost
```


```python
# minimize the cost function
@tf.function
def train(X, Y, learning_rate = 0.1):
    with tf.GradientTape() as tape:
        _loss = loss(Hypothesis(X), Y)
    _w, _b = tape.gradient(_loss, [W, B])
    W.assign_sub(learning_rate * _w)
    B.assign_sub(learning_rate * _b)
```


```python
# accuracy computation

# tf.argmax : 2차원 배열의 각 행에서 가장 큰 값을 리턴 시켜줌
@tf.function
def evaluation(H, Y):
    prediction = tf.argmax(H, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return prediction, accuracy
```


```python
# training

training_epochs = 50
batch_size = 100

for epoch in range(training_epochs):
    avg_cost = 0
    iterations = int(len(x_train) / batch_size) 
    
    idx = 0
    for i in range(iterations):
        batch_xs, batch_ys = x_train[idx:idx+batch_size, :], y_train[idx:idx+batch_size, :]
        _c = loss(Hypothesis(batch_xs), batch_ys)
        train(batch_xs, batch_ys, learning_rate=0.15)
        
        avg_cost += _c / iterations
        idx += batch_size
    print("epoch: {:2d} loss: {}".format(epoch+1, avg_cost))
```

    epoch:  1 loss: 2.304088830947876
    epoch:  2 loss: 0.9827862977981567
    epoch:  3 loss: 0.789933443069458
    epoch:  4 loss: 0.6923120617866516
    epoch:  5 loss: 0.6300580501556396
    epoch:  6 loss: 0.5856894254684448
    epoch:  7 loss: 0.5519533753395081
    epoch:  8 loss: 0.5251486301422119
    epoch:  9 loss: 0.5031442642211914
    epoch: 10 loss: 0.48462745547294617
    epoch: 11 loss: 0.4687447249889374
    epoch: 12 loss: 0.4549160301685333
    epoch: 13 loss: 0.44272923469543457
    epoch: 14 loss: 0.4318826198577881
    epoch: 15 loss: 0.42214906215667725
    epoch: 16 loss: 0.41335123777389526
    epoch: 17 loss: 0.4053502380847931
    epoch: 18 loss: 0.398034006357193
    epoch: 19 loss: 0.3913104236125946
    epoch: 20 loss: 0.3851042687892914
    epoch: 21 loss: 0.379353404045105
    epoch: 22 loss: 0.3740047514438629
    epoch: 23 loss: 0.36901360750198364
    epoch: 24 loss: 0.36434271931648254
    epoch: 25 loss: 0.35996004939079285
    epoch: 26 loss: 0.35583731532096863
    epoch: 27 loss: 0.35195040702819824
    epoch: 28 loss: 0.3482787013053894
    epoch: 29 loss: 0.34480345249176025
    epoch: 30 loss: 0.3415086269378662
    epoch: 31 loss: 0.33838003873825073
    epoch: 32 loss: 0.33540478348731995
    epoch: 33 loss: 0.3325716257095337
    epoch: 34 loss: 0.3298702538013458
    epoch: 35 loss: 0.32729172706604004
    epoch: 36 loss: 0.3248271644115448
    epoch: 37 loss: 0.32247018814086914
    epoch: 38 loss: 0.3202127516269684
    epoch: 39 loss: 0.31804919242858887
    epoch: 40 loss: 0.3159739673137665
    epoch: 41 loss: 0.3139815032482147
    epoch: 42 loss: 0.31206679344177246
    epoch: 43 loss: 0.31022587418556213
    epoch: 44 loss: 0.30845460295677185
    epoch: 45 loss: 0.30674904584884644
    epoch: 46 loss: 0.3051054775714874
    epoch: 47 loss: 0.30352064967155457
    epoch: 48 loss: 0.30199187994003296
    epoch: 49 loss: 0.30051612854003906
    epoch: 50 loss: 0.29909056425094604
    


```python
# report accuracy
print("Accuracy ....")
_h = Hypothesis(x_test)
_p, _a = evaluation(_h, y_test)
print("Accuracy : ", _a.numpy)
```

    Accuracy ....
    Accuracy :  <bound method _EagerTensorBase.numpy of <tf.Tensor: shape=(), dtype=float32, numpy=0.9127>>
    

## C. Classification for MNIST using Keras
- Keras는 신경망을 구성하기 위한 각 구성요소를 클래스로 제공
- tf.keras는 tensorflow의 high-level API

#### Keras 사용의 일반적인 절차

1) model 구성
- Keras의 Sequential() 클래스로 model 객체 생성
- model에 필요한 layer를 추가

2) model.compile() : 모델의 학습과정 설정
- optimizer와 loss함수 설정

3) model.fit() : 학습
- batch_size, epochs 등을 설정
- loss, accuracy 측정

4) model.evaluate() : 성능평가
- 준비된 test dataset으로 학습한 모델 평가

5) model.predict() : 모델사용
- 임의의 입력 데이터에 대한 model의 예측결과 얻기


```python
import tensorflow as tf
import matplotlib.pyplot as plt
```


```python
mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

plt.figure(figsize=(8, 2)) # 8 x 2 inchs
for i in range(36):
    plt.subplot(3, 12, i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.axis("off")
plt.show()
```


    
![output_39_0](https://user-images.githubusercontent.com/86539195/135233589-09839e98-5853-4a01-ba2c-19cb95d0c78a.png)
    



```python
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)
```

    (60000, 28, 28) float64
    (60000,) uint8
    (10000, 28, 28) float64
    (10000,) uint8
    

- model 구성


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, input_dim=784, activation='softmax')
])
```

- model.compile() : 모델의 학습과정 설정


```python
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                7850      
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________
    

- model.fit() : 학습


```python
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=2, batch_size=100, epochs=15, use_multiprocessing=True)
model.evaluate(X_test, Y_test, verbose=2, batch_size=100, use_multiprocessing=True)
```

    Epoch 1/15
    600/600 - 3s - loss: 0.6248 - accuracy: 0.8444 - val_loss: 0.3618 - val_accuracy: 0.9064
    Epoch 2/15
    600/600 - 2s - loss: 0.3463 - accuracy: 0.9055 - val_loss: 0.3080 - val_accuracy: 0.9159
    Epoch 3/15
    600/600 - 2s - loss: 0.3096 - accuracy: 0.9147 - val_loss: 0.2922 - val_accuracy: 0.9177
    Epoch 4/15
    600/600 - 1s - loss: 0.2927 - accuracy: 0.9182 - val_loss: 0.2808 - val_accuracy: 0.9230
    Epoch 5/15
    600/600 - 1s - loss: 0.2822 - accuracy: 0.9213 - val_loss: 0.2753 - val_accuracy: 0.9244
    Epoch 6/15
    600/600 - 1s - loss: 0.2751 - accuracy: 0.9232 - val_loss: 0.2736 - val_accuracy: 0.9228
    Epoch 7/15
    600/600 - 1s - loss: 0.2698 - accuracy: 0.9249 - val_loss: 0.2721 - val_accuracy: 0.9247
    Epoch 8/15
    600/600 - 2s - loss: 0.2657 - accuracy: 0.9256 - val_loss: 0.2705 - val_accuracy: 0.9234
    Epoch 9/15
    600/600 - 2s - loss: 0.2624 - accuracy: 0.9267 - val_loss: 0.2651 - val_accuracy: 0.9274
    Epoch 10/15
    600/600 - 1s - loss: 0.2592 - accuracy: 0.9279 - val_loss: 0.2660 - val_accuracy: 0.9265
    Epoch 11/15
    600/600 - 1s - loss: 0.2568 - accuracy: 0.9282 - val_loss: 0.2619 - val_accuracy: 0.9277
    Epoch 12/15
    600/600 - 1s - loss: 0.2551 - accuracy: 0.9293 - val_loss: 0.2670 - val_accuracy: 0.9262
    Epoch 13/15
    600/600 - 1s - loss: 0.2531 - accuracy: 0.9295 - val_loss: 0.2656 - val_accuracy: 0.9261
    Epoch 14/15
    600/600 - 1s - loss: 0.2515 - accuracy: 0.9305 - val_loss: 0.2644 - val_accuracy: 0.9254
    Epoch 15/15
    600/600 - 1s - loss: 0.2500 - accuracy: 0.9312 - val_loss: 0.2647 - val_accuracy: 0.9267
    100/100 - 0s - loss: 0.2647 - accuracy: 0.9267
    




    [0.26471731066703796, 0.9266999959945679]



- model.evaluate() : 성능평가


```python
plt.figure(figsize=(8, 4)) # 8 x 4 inchs
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("Cost Graph")
plt.ylabel("cost")
plt.subplot(1, 2, 2)
plt.title("Accuracy Graph")
plt.ylabel("accuracy")
plt.plot(hist.history['accuracy'], 'b-', label="training accuracy")
plt.plot(hist.history['val_accuracy'], 'r:', label="validation accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Accuracy Graph를 보면 overfitting 되지 않음
```


    
![output_48_0](https://user-images.githubusercontent.com/86539195/135233630-8bd3c70b-cb3c-427a-9cbb-564d1a69a941.png)
    


- model.predict() : 모델사용


```python
prediction = model.predict(X_test[:1, :])
prediction_class = tf.argmax(prediction, 1)

print(f"\nPrediction Result:\n{prediction}")
print("Predicted class: ", prediction_class.numpy())
plt.imshow(X_test[prediction_class[0]])
plt.axis("off")
plt.show()

# 아래 결과는 9를 7로 예측
```

    
    Prediction Result:
    [[1.55858822e-06 1.28604115e-11 7.45132456e-06 3.46772699e-03
      2.17767592e-07 2.67602663e-05 9.74907932e-11 9.96214092e-01
      1.25328979e-05 2.69677956e-04]]
    Predicted class:  [7]
    


    
![output_50_1](https://user-images.githubusercontent.com/86539195/135233659-76745b77-4b51-4d7f-8a18-0248d2766b5e.png)
    



```python
prediction = model.predict(X_test[:20, :])
prediction_class = tf.argmax(prediction, 1)

print(f"\nPrediction Result:\n{prediction}")
print("Predicted class: ", prediction_class.numpy())

plt.figure(figsize=(8, 2)) # 8 x 2 inchs
for i in range(20):
    plt.subplot(3, 12, i+1)
    plt.imshow(X_test[prediction_class[i]])
    plt.axis("off")
plt.show()
```

    
    Prediction Result:
    [[1.55858822e-06 1.28604610e-11 7.45132456e-06 3.46772699e-03
      2.17767592e-07 2.67602663e-05 9.74907932e-11 9.96214092e-01
      1.25328979e-05 2.69677956e-04]
     [2.20619870e-04 2.72424245e-06 9.93943393e-01 7.25481150e-05
      1.12366383e-14 2.47653783e-03 3.25564877e-03 1.12229136e-17
      2.84528724e-05 4.30808479e-14]
     [1.21852224e-06 9.82354820e-01 1.05629610e-02 1.94879551e-03
      1.04412771e-04 6.86221698e-04 1.00970501e-03 7.28404731e-04
      2.43330095e-03 1.70174593e-04]
     [9.99800026e-01 1.56038914e-11 5.57947060e-05 1.17326579e-06
      1.41874335e-08 3.97002877e-05 8.07294855e-05 7.17776720e-06
      7.20489788e-06 8.25036386e-06]
     [4.52065375e-04 2.51975933e-07 2.60784035e-03 2.12344330e-05
      9.59268034e-01 1.29483204e-04 1.81214232e-03 3.78520344e-03
      3.39230336e-03 2.85315011e-02]
     [7.27158849e-08 9.92971957e-01 1.73433544e-03 8.37349158e-04
      8.25329334e-06 3.61239618e-05 1.46491293e-05 2.82020192e-03
      1.33076089e-03 2.46295967e-04]
     [1.06430196e-06 6.61802204e-08 5.67574091e-07 6.24762761e-05
      9.78148997e-01 3.34225246e-03 1.80866209e-05 1.17703108e-03
      6.84536668e-03 1.04040932e-02]
     [1.03563842e-08 8.04778282e-03 1.61478616e-04 2.97833147e-04
      5.25017316e-03 5.63124986e-03 8.51242112e-06 3.15063284e-04
      1.12549728e-03 9.79162455e-01]
     [6.85513151e-05 5.13207477e-10 4.09886479e-06 3.52450291e-10
      1.68248516e-04 2.45426374e-04 9.99479234e-01 8.32801467e-11
      3.44784785e-05 2.78039245e-08]
     [1.40676548e-07 4.27019219e-11 1.42878323e-08 6.79377536e-07
      1.19704921e-02 6.41470660e-06 1.93582608e-08 4.88134809e-02
      9.48509725e-04 9.38260257e-01]
     [9.90717471e-01 1.99188790e-10 1.24401285e-03 4.63462602e-05
      1.43707425e-07 6.81428518e-03 3.23854656e-05 2.32650432e-08
      1.14452990e-03 7.30719535e-07]
     [1.73438701e-03 3.19784653e-04 3.63496430e-02 1.92100939e-03
      7.93308602e-04 1.97020170e-04 9.20109808e-01 1.81992095e-06
      3.85369360e-02 3.63269828e-05]
     [4.80333256e-06 3.82774665e-08 2.80460772e-05 5.20900823e-04
      6.88122120e-03 5.13224746e-04 2.69818997e-06 2.65357476e-02
      1.62371190e-03 9.63889599e-01]
     [9.98615384e-01 8.86302282e-11 7.28524974e-05 3.42499493e-06
      9.02454133e-07 3.42539046e-04 2.02213201e-07 1.80636871e-05
      6.12790463e-04 3.33764037e-04]
     [5.65051561e-10 9.98103380e-01 7.80031114e-05 1.35203090e-03
      6.22846130e-08 1.55572834e-05 1.90312676e-05 4.87535772e-06
      3.67730798e-04 5.93088807e-05]
     [5.30534540e-04 1.29943965e-05 2.20358619e-04 3.78042497e-02
      2.68755284e-05 9.10369217e-01 3.68130459e-05 5.20996707e-07
      5.09967357e-02 1.60489128e-06]
     [7.65859222e-05 1.31052913e-09 6.06516027e-04 5.47523377e-05
      1.80482175e-02 7.54079156e-05 1.18918588e-05 3.72945592e-02
      6.86328160e-03 9.36968863e-01]
     [1.24129792e-06 2.69500761e-14 5.27124894e-06 4.53108409e-03
      1.24723210e-08 6.31830915e-07 3.41990013e-11 9.95458245e-01
      5.09387860e-07 3.06565676e-06]
     [1.63352786e-04 5.02497278e-06 1.76079258e-01 6.50925517e-01
      1.32976973e-04 9.65633392e-02 1.95876230e-02 2.10158287e-05
      5.65199181e-02 1.95259850e-06]
     [2.46384789e-05 1.85390127e-06 5.56055893e-05 1.13605580e-04
      9.82152045e-01 4.65266319e-04 1.55280257e-04 3.72439739e-04
      3.70929367e-04 1.62883718e-02]]
    Predicted class:  [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
    


    
![output_51_1](https://user-images.githubusercontent.com/86539195/135233710-eb4aa1bb-e6de-41c0-af9a-100dbc54929e.png)
    



```python
# save model

file_name = "softmax_mnist_model.h5"
model.save(file_name)
print(f"\nThis model has been saved to {file_name}.")
```

    
    This model has been saved to softmax_mnist_model.h5.
    

- 위에서 저장한 모델을 불러와서 학습시키기


```python
# 가중치와 옵티마이저를 포함하여 정확히 동일한 전체 모델을 HDF5 파일로 부터 로딩
file_name = "softmax_mnist_model.h5"
model = tf.keras.models.load_model(file_name)
print(f"\nThis model has been loaded from {file_name}.")

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])

hist = model.fit(X_train, Y_train,validation_data=(X_test, Y_test),verbose=2, batch_size=100, epochs=15, use_multiprocessing=True)

model.evaluate(X_test, Y_test,verbose=2, batch_size=100, use_multiprocessing=True)
```

    
    This model has been loaded from softmax_mnist_model.h5.
    Epoch 1/15
    600/600 - 3s - loss: 0.2489 - accuracy: 0.9308 - val_loss: 0.2632 - val_accuracy: 0.9277
    Epoch 2/15
    600/600 - 2s - loss: 0.2476 - accuracy: 0.9314 - val_loss: 0.2653 - val_accuracy: 0.9262
    Epoch 3/15
    600/600 - 1s - loss: 0.2462 - accuracy: 0.9317 - val_loss: 0.2614 - val_accuracy: 0.9278
    Epoch 4/15
    600/600 - 1s - loss: 0.2450 - accuracy: 0.9324 - val_loss: 0.2621 - val_accuracy: 0.9281
    Epoch 5/15
    600/600 - 2s - loss: 0.2442 - accuracy: 0.9330 - val_loss: 0.2643 - val_accuracy: 0.9271
    Epoch 6/15
    600/600 - 1s - loss: 0.2433 - accuracy: 0.9328 - val_loss: 0.2606 - val_accuracy: 0.9275
    Epoch 7/15
    600/600 - 1s - loss: 0.2428 - accuracy: 0.9336 - val_loss: 0.2646 - val_accuracy: 0.9264
    Epoch 8/15
    600/600 - 2s - loss: 0.2416 - accuracy: 0.9336 - val_loss: 0.2643 - val_accuracy: 0.9276
    Epoch 9/15
    600/600 - 2s - loss: 0.2410 - accuracy: 0.9328 - val_loss: 0.2637 - val_accuracy: 0.9276
    Epoch 10/15
    600/600 - 1s - loss: 0.2403 - accuracy: 0.9339 - val_loss: 0.2621 - val_accuracy: 0.9276
    Epoch 11/15
    600/600 - 1s - loss: 0.2396 - accuracy: 0.9346 - val_loss: 0.2644 - val_accuracy: 0.9278
    Epoch 12/15
    600/600 - 2s - loss: 0.2391 - accuracy: 0.9347 - val_loss: 0.2652 - val_accuracy: 0.9270
    Epoch 13/15
    600/600 - 1s - loss: 0.2385 - accuracy: 0.9346 - val_loss: 0.2626 - val_accuracy: 0.9274
    Epoch 14/15
    600/600 - 1s - loss: 0.2379 - accuracy: 0.9347 - val_loss: 0.2633 - val_accuracy: 0.9271
    Epoch 15/15
    600/600 - 1s - loss: 0.2377 - accuracy: 0.9348 - val_loss: 0.2655 - val_accuracy: 0.9276
    100/100 - 0s - loss: 0.2655 - accuracy: 0.9276
    




    [0.26552286744117737, 0.9276000261306763]




```python
plt.figure(figsize=(8, 4)) # 8 x 4 inchs
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("Cost Graph")
plt.ylabel("cost")
plt.subplot(1, 2, 2)
plt.title("Accuracy Graph")
plt.ylabel("accuracy")
plt.plot(hist.history['accuracy'], 'b-', label="training accuracy")
plt.plot(hist.history['val_accuracy'], 'r:', label="validation accuracy")
plt.legend()
plt.tight_layout()
plt.show()
```


    
![output_55_0](https://user-images.githubusercontent.com/86539195/135233737-c50783df-b1d8-4839-898f-5af55f365e9c.png)
    



```python
prediction = model.predict(X_test[:20, :])
prediction_class = tf.argmax(prediction, 1)

print(f"\nPrediction Result:\n{prediction}")
print("Predicted class: ", prediction_class.numpy())

plt.figure(figsize=(8, 2)) # 8 x 2 inchs
for i in range(20):
    plt.subplot(3, 12, i+1)
    plt.imshow(X_test[prediction_class[i]])
    plt.axis("off")
plt.show()
```

    
    Prediction Result:
    [[1.67824865e-07 2.13619514e-13 8.73152999e-07 4.88947006e-03
      1.64069988e-07 2.12577434e-05 6.56449992e-13 9.94794786e-01
      1.05818053e-05 2.82797497e-04]
     [3.63977051e-05 8.07713946e-07 9.98028457e-01 4.49703157e-06
      7.65164897e-17 7.45357072e-04 1.17910071e-03 1.69392695e-21
      5.39986650e-06 2.65434318e-18]
     [4.81173345e-07 9.82919335e-01 1.21030454e-02 1.46890339e-03
      7.52687920e-05 6.32718264e-04 6.09273033e-04 7.19488162e-05
      1.97838107e-03 1.40710763e-04]
     [9.99912143e-01 3.34255388e-12 4.20263641e-05 4.96276584e-07
      5.62648417e-09 8.13994484e-06 3.25869587e-05 1.15034959e-06
      1.36091808e-06 2.08083998e-06]
     [5.75272716e-04 4.13668886e-08 2.04670290e-03 1.22525853e-05
      9.72229719e-01 7.30390529e-05 1.44349271e-03 2.11214460e-03
      1.62602065e-03 1.98813584e-02]
     [3.95685120e-08 9.92426634e-01 2.16250937e-03 7.33251160e-04
      5.19645300e-06 2.70328492e-05 5.86750548e-06 3.18999356e-03
      1.22472201e-03 2.24804477e-04]
     [6.76546165e-07 1.17380727e-08 2.84330525e-07 5.98254264e-05
      9.86333907e-01 1.78762758e-03 7.21550759e-06 8.06817610e-04
      4.60449932e-03 6.39913743e-03]
     [1.69853265e-09 1.02537079e-02 9.68565437e-05 2.38481109e-04
      4.71952651e-03 5.36450744e-03 3.83537611e-07 2.05342949e-04
      4.51707107e-04 9.78669524e-01]
     [1.27878971e-04 3.28057477e-11 2.36819915e-06 2.57169508e-10
      1.09386317e-04 2.40355061e-04 9.99490619e-01 7.55702033e-13
      2.93850517e-05 3.10537152e-10]
     [4.39124754e-08 1.05814923e-12 1.39145173e-09 6.55824920e-07
      1.54346572e-02 1.45415254e-06 1.61281492e-11 2.91143078e-02
      7.93128740e-04 9.54655766e-01]
     [9.93178368e-01 1.08203058e-10 1.08177389e-03 4.17256706e-05
      1.13336966e-07 5.09455707e-03 3.09927818e-05 2.92937963e-09
      5.71715645e-04 7.28364739e-07]
     [1.53267395e-03 5.41994348e-04 4.10080031e-02 9.85393999e-04
      4.28837899e-04 6.52227100e-05 9.28812265e-01 9.17371850e-08
      2.66239773e-02 1.49735581e-06]
     [1.42255317e-06 8.66893224e-09 1.23839391e-05 1.09091122e-03
      7.01147504e-03 4.44236852e-04 1.97941191e-07 1.85490735e-02
      1.37483049e-03 9.71515536e-01]
     [9.99373496e-01 3.30386239e-11 4.73138316e-05 2.67715654e-06
      5.88678517e-07 1.59847521e-04 8.99037218e-08 1.16552510e-05
      2.26482167e-04 1.77828086e-04]
     [1.74814080e-10 9.98066366e-01 1.06025676e-04 1.42218813e-03
      2.00229149e-08 9.56980602e-06 7.22496316e-06 2.88304977e-06
      3.30577663e-04 5.51512676e-05]
     [2.82187335e-04 5.35969548e-06 1.95548258e-04 6.07497357e-02
      8.84319434e-06 8.82978380e-01 4.31468397e-05 3.20330695e-09
      5.57360090e-02 7.24323115e-07]
     [6.49552239e-05 1.44566845e-10 9.91324894e-04 6.45323962e-05
      2.04110965e-02 6.00345847e-05 1.62147012e-06 2.39421409e-02
      6.12644525e-03 9.48337853e-01]
     [2.58893181e-07 2.45192321e-16 1.62417371e-06 9.37283412e-03
      1.63431828e-08 5.62398952e-07 7.32926686e-13 9.90621686e-01
      4.36536737e-07 2.66800612e-06]
     [1.17943760e-04 2.03514773e-07 2.74367094e-01 6.30569816e-01
      4.11826950e-05 3.71568054e-02 1.52311614e-02 1.43867612e-06
      4.25143540e-02 2.29948238e-08]
     [1.88386257e-05 1.20674690e-06 2.66504339e-05 8.89824078e-05
      9.90507841e-01 2.89913150e-04 7.03261248e-05 1.63655728e-04
      1.67895632e-04 8.66470207e-03]]
    Predicted class:  [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
    


    
![output_56_1](https://user-images.githubusercontent.com/86539195/135233759-79060330-5a0f-4663-ae7b-4b9704da6c7b.png)
    


>NN for XOR(Tensorflow 구현)


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```


```python
# XOR data
x_data = tf.constant([[0, 0],
[0, 1],
[1, 0],
[1, 1]], dtype=tf.float32)

y_data = tf.constant([[0],
[1],
[1],
[0]], dtype=tf.float32)

print(x_data.shape, x_data.dtype)
print(y_data.shape, y_data.dtype)
```

    (4, 2) <dtype: 'float32'>
    (4, 1) <dtype: 'float32'>
    


```python
model = tf.keras.models.Sequential([tf.keras.layers.Dense(8, input_dim=2, activation='relu'),tf.keras.layers.Dense(1, activation='sigmoid')])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_2 (Dense)              (None, 8)                 24        
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 9         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer='adam',loss='mean_squared_error',metrics=["accuracy"])
model.summary()

hist = model.fit(x_data, y_data, batch_size=4,epochs=100, validation_data=(x_data, y_data),verbose=2, use_multiprocessing=True)

model.evaluate(x_data, y_data, verbose=2, use_multiprocessing=True)
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_2 (Dense)              (None, 8)                 24        
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 9         
    =================================================================
    Total params: 33
    Trainable params: 33
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    1/1 - 1s - loss: 0.2580 - accuracy: 0.5000 - val_loss: 0.2577 - val_accuracy: 0.2500
    Epoch 2/100
    1/1 - 0s - loss: 0.2577 - accuracy: 0.2500 - val_loss: 0.2574 - val_accuracy: 0.2500
    Epoch 3/100
    1/1 - 0s - loss: 0.2574 - accuracy: 0.2500 - val_loss: 0.2571 - val_accuracy: 0.2500
    Epoch 4/100
    1/1 - 0s - loss: 0.2571 - accuracy: 0.2500 - val_loss: 0.2568 - val_accuracy: 0.2500
    Epoch 5/100
    1/1 - 0s - loss: 0.2568 - accuracy: 0.2500 - val_loss: 0.2565 - val_accuracy: 0.2500
    Epoch 6/100
    1/1 - 0s - loss: 0.2565 - accuracy: 0.2500 - val_loss: 0.2561 - val_accuracy: 0.2500
    Epoch 7/100
    1/1 - 0s - loss: 0.2561 - accuracy: 0.2500 - val_loss: 0.2558 - val_accuracy: 0.2500
    Epoch 8/100
    1/1 - 0s - loss: 0.2558 - accuracy: 0.2500 - val_loss: 0.2555 - val_accuracy: 0.2500
    Epoch 9/100
    1/1 - 0s - loss: 0.2555 - accuracy: 0.2500 - val_loss: 0.2552 - val_accuracy: 0.2500
    Epoch 10/100
    1/1 - 0s - loss: 0.2552 - accuracy: 0.2500 - val_loss: 0.2549 - val_accuracy: 0.2500
    Epoch 11/100
    1/1 - 0s - loss: 0.2549 - accuracy: 0.2500 - val_loss: 0.2546 - val_accuracy: 0.2500
    Epoch 12/100
    1/1 - 0s - loss: 0.2546 - accuracy: 0.2500 - val_loss: 0.2544 - val_accuracy: 0.2500
    Epoch 13/100
    1/1 - 0s - loss: 0.2544 - accuracy: 0.2500 - val_loss: 0.2541 - val_accuracy: 0.2500
    Epoch 14/100
    1/1 - 0s - loss: 0.2541 - accuracy: 0.2500 - val_loss: 0.2538 - val_accuracy: 0.2500
    Epoch 15/100
    1/1 - 0s - loss: 0.2538 - accuracy: 0.2500 - val_loss: 0.2535 - val_accuracy: 0.2500
    Epoch 16/100
    1/1 - 0s - loss: 0.2535 - accuracy: 0.2500 - val_loss: 0.2532 - val_accuracy: 0.2500
    Epoch 17/100
    1/1 - 0s - loss: 0.2532 - accuracy: 0.2500 - val_loss: 0.2530 - val_accuracy: 0.2500
    Epoch 18/100
    1/1 - 0s - loss: 0.2530 - accuracy: 0.2500 - val_loss: 0.2527 - val_accuracy: 0.2500
    Epoch 19/100
    1/1 - 0s - loss: 0.2527 - accuracy: 0.2500 - val_loss: 0.2524 - val_accuracy: 0.2500
    Epoch 20/100
    1/1 - 0s - loss: 0.2524 - accuracy: 0.2500 - val_loss: 0.2521 - val_accuracy: 0.2500
    Epoch 21/100
    1/1 - 0s - loss: 0.2521 - accuracy: 0.2500 - val_loss: 0.2519 - val_accuracy: 0.2500
    Epoch 22/100
    1/1 - 0s - loss: 0.2519 - accuracy: 0.2500 - val_loss: 0.2516 - val_accuracy: 0.2500
    Epoch 23/100
    1/1 - 0s - loss: 0.2516 - accuracy: 0.2500 - val_loss: 0.2514 - val_accuracy: 0.5000
    Epoch 24/100
    1/1 - 0s - loss: 0.2514 - accuracy: 0.5000 - val_loss: 0.2511 - val_accuracy: 0.5000
    Epoch 25/100
    1/1 - 0s - loss: 0.2511 - accuracy: 0.5000 - val_loss: 0.2509 - val_accuracy: 0.5000
    Epoch 26/100
    1/1 - 0s - loss: 0.2509 - accuracy: 0.5000 - val_loss: 0.2506 - val_accuracy: 0.5000
    Epoch 27/100
    1/1 - 0s - loss: 0.2506 - accuracy: 0.5000 - val_loss: 0.2504 - val_accuracy: 0.5000
    Epoch 28/100
    1/1 - 0s - loss: 0.2504 - accuracy: 0.5000 - val_loss: 0.2501 - val_accuracy: 0.5000
    Epoch 29/100
    1/1 - 0s - loss: 0.2501 - accuracy: 0.5000 - val_loss: 0.2499 - val_accuracy: 0.5000
    Epoch 30/100
    1/1 - 0s - loss: 0.2499 - accuracy: 0.5000 - val_loss: 0.2496 - val_accuracy: 0.5000
    Epoch 31/100
    1/1 - 0s - loss: 0.2496 - accuracy: 0.5000 - val_loss: 0.2494 - val_accuracy: 0.5000
    Epoch 32/100
    1/1 - 0s - loss: 0.2494 - accuracy: 0.5000 - val_loss: 0.2492 - val_accuracy: 0.5000
    Epoch 33/100
    1/1 - 0s - loss: 0.2492 - accuracy: 0.5000 - val_loss: 0.2489 - val_accuracy: 0.5000
    Epoch 34/100
    1/1 - 0s - loss: 0.2489 - accuracy: 0.5000 - val_loss: 0.2487 - val_accuracy: 0.5000
    Epoch 35/100
    1/1 - 0s - loss: 0.2487 - accuracy: 0.5000 - val_loss: 0.2485 - val_accuracy: 0.5000
    Epoch 36/100
    1/1 - 0s - loss: 0.2485 - accuracy: 0.5000 - val_loss: 0.2482 - val_accuracy: 0.5000
    Epoch 37/100
    1/1 - 0s - loss: 0.2482 - accuracy: 0.5000 - val_loss: 0.2480 - val_accuracy: 0.5000
    Epoch 38/100
    1/1 - 0s - loss: 0.2480 - accuracy: 0.5000 - val_loss: 0.2478 - val_accuracy: 0.5000
    Epoch 39/100
    1/1 - 0s - loss: 0.2478 - accuracy: 0.5000 - val_loss: 0.2475 - val_accuracy: 0.5000
    Epoch 40/100
    1/1 - 0s - loss: 0.2475 - accuracy: 0.5000 - val_loss: 0.2473 - val_accuracy: 0.5000
    Epoch 41/100
    1/1 - 0s - loss: 0.2473 - accuracy: 0.5000 - val_loss: 0.2471 - val_accuracy: 0.5000
    Epoch 42/100
    1/1 - 0s - loss: 0.2471 - accuracy: 0.5000 - val_loss: 0.2469 - val_accuracy: 0.5000
    Epoch 43/100
    1/1 - 0s - loss: 0.2469 - accuracy: 0.5000 - val_loss: 0.2467 - val_accuracy: 0.5000
    Epoch 44/100
    1/1 - 0s - loss: 0.2467 - accuracy: 0.5000 - val_loss: 0.2464 - val_accuracy: 0.5000
    Epoch 45/100
    1/1 - 0s - loss: 0.2464 - accuracy: 0.5000 - val_loss: 0.2462 - val_accuracy: 0.5000
    Epoch 46/100
    1/1 - 0s - loss: 0.2462 - accuracy: 0.5000 - val_loss: 0.2460 - val_accuracy: 0.5000
    Epoch 47/100
    1/1 - 0s - loss: 0.2460 - accuracy: 0.5000 - val_loss: 0.2458 - val_accuracy: 0.5000
    Epoch 48/100
    1/1 - 0s - loss: 0.2458 - accuracy: 0.5000 - val_loss: 0.2456 - val_accuracy: 0.5000
    Epoch 49/100
    1/1 - 0s - loss: 0.2456 - accuracy: 0.5000 - val_loss: 0.2454 - val_accuracy: 0.5000
    Epoch 50/100
    1/1 - 0s - loss: 0.2454 - accuracy: 0.5000 - val_loss: 0.2451 - val_accuracy: 0.5000
    Epoch 51/100
    1/1 - 0s - loss: 0.2451 - accuracy: 0.5000 - val_loss: 0.2449 - val_accuracy: 0.5000
    Epoch 52/100
    1/1 - 0s - loss: 0.2449 - accuracy: 0.5000 - val_loss: 0.2447 - val_accuracy: 0.5000
    Epoch 53/100
    1/1 - 0s - loss: 0.2447 - accuracy: 0.5000 - val_loss: 0.2445 - val_accuracy: 0.5000
    Epoch 54/100
    1/1 - 0s - loss: 0.2445 - accuracy: 0.5000 - val_loss: 0.2443 - val_accuracy: 0.5000
    Epoch 55/100
    1/1 - 0s - loss: 0.2443 - accuracy: 0.5000 - val_loss: 0.2441 - val_accuracy: 0.5000
    Epoch 56/100
    1/1 - 0s - loss: 0.2441 - accuracy: 0.5000 - val_loss: 0.2439 - val_accuracy: 0.5000
    Epoch 57/100
    1/1 - 0s - loss: 0.2439 - accuracy: 0.5000 - val_loss: 0.2437 - val_accuracy: 0.5000
    Epoch 58/100
    1/1 - 0s - loss: 0.2437 - accuracy: 0.5000 - val_loss: 0.2435 - val_accuracy: 0.5000
    Epoch 59/100
    1/1 - 0s - loss: 0.2435 - accuracy: 0.5000 - val_loss: 0.2433 - val_accuracy: 0.5000
    Epoch 60/100
    1/1 - 0s - loss: 0.2433 - accuracy: 0.5000 - val_loss: 0.2431 - val_accuracy: 0.5000
    Epoch 61/100
    1/1 - 0s - loss: 0.2431 - accuracy: 0.5000 - val_loss: 0.2429 - val_accuracy: 0.5000
    Epoch 62/100
    1/1 - 0s - loss: 0.2429 - accuracy: 0.5000 - val_loss: 0.2427 - val_accuracy: 0.5000
    Epoch 63/100
    1/1 - 0s - loss: 0.2427 - accuracy: 0.5000 - val_loss: 0.2425 - val_accuracy: 0.5000
    Epoch 64/100
    1/1 - 0s - loss: 0.2425 - accuracy: 0.5000 - val_loss: 0.2423 - val_accuracy: 0.5000
    Epoch 65/100
    1/1 - 0s - loss: 0.2423 - accuracy: 0.5000 - val_loss: 0.2421 - val_accuracy: 0.5000
    Epoch 66/100
    1/1 - 0s - loss: 0.2421 - accuracy: 0.5000 - val_loss: 0.2419 - val_accuracy: 0.5000
    Epoch 67/100
    1/1 - 0s - loss: 0.2419 - accuracy: 0.5000 - val_loss: 0.2417 - val_accuracy: 0.5000
    Epoch 68/100
    1/1 - 0s - loss: 0.2417 - accuracy: 0.5000 - val_loss: 0.2415 - val_accuracy: 0.5000
    Epoch 69/100
    1/1 - 0s - loss: 0.2415 - accuracy: 0.5000 - val_loss: 0.2413 - val_accuracy: 0.5000
    Epoch 70/100
    1/1 - 0s - loss: 0.2413 - accuracy: 0.5000 - val_loss: 0.2411 - val_accuracy: 0.5000
    Epoch 71/100
    1/1 - 0s - loss: 0.2411 - accuracy: 0.5000 - val_loss: 0.2409 - val_accuracy: 0.5000
    Epoch 72/100
    1/1 - 0s - loss: 0.2409 - accuracy: 0.5000 - val_loss: 0.2407 - val_accuracy: 0.5000
    Epoch 73/100
    1/1 - 0s - loss: 0.2407 - accuracy: 0.5000 - val_loss: 0.2405 - val_accuracy: 0.5000
    Epoch 74/100
    1/1 - 0s - loss: 0.2405 - accuracy: 0.5000 - val_loss: 0.2403 - val_accuracy: 0.5000
    Epoch 75/100
    1/1 - 0s - loss: 0.2403 - accuracy: 0.5000 - val_loss: 0.2401 - val_accuracy: 0.5000
    Epoch 76/100
    1/1 - 0s - loss: 0.2401 - accuracy: 0.5000 - val_loss: 0.2399 - val_accuracy: 0.5000
    Epoch 77/100
    1/1 - 0s - loss: 0.2399 - accuracy: 0.5000 - val_loss: 0.2397 - val_accuracy: 0.5000
    Epoch 78/100
    1/1 - 0s - loss: 0.2397 - accuracy: 0.5000 - val_loss: 0.2395 - val_accuracy: 0.5000
    Epoch 79/100
    1/1 - 0s - loss: 0.2395 - accuracy: 0.5000 - val_loss: 0.2393 - val_accuracy: 0.5000
    Epoch 80/100
    1/1 - 0s - loss: 0.2393 - accuracy: 0.5000 - val_loss: 0.2391 - val_accuracy: 0.5000
    Epoch 81/100
    1/1 - 0s - loss: 0.2391 - accuracy: 0.5000 - val_loss: 0.2390 - val_accuracy: 0.5000
    Epoch 82/100
    1/1 - 0s - loss: 0.2390 - accuracy: 0.5000 - val_loss: 0.2388 - val_accuracy: 0.5000
    Epoch 83/100
    1/1 - 0s - loss: 0.2388 - accuracy: 0.5000 - val_loss: 0.2386 - val_accuracy: 0.5000
    Epoch 84/100
    1/1 - 0s - loss: 0.2386 - accuracy: 0.5000 - val_loss: 0.2384 - val_accuracy: 0.5000
    Epoch 85/100
    1/1 - 0s - loss: 0.2384 - accuracy: 0.5000 - val_loss: 0.2382 - val_accuracy: 0.5000
    Epoch 86/100
    1/1 - 0s - loss: 0.2382 - accuracy: 0.5000 - val_loss: 0.2380 - val_accuracy: 0.5000
    Epoch 87/100
    1/1 - 0s - loss: 0.2380 - accuracy: 0.5000 - val_loss: 0.2378 - val_accuracy: 0.5000
    Epoch 88/100
    1/1 - 0s - loss: 0.2378 - accuracy: 0.5000 - val_loss: 0.2376 - val_accuracy: 0.5000
    Epoch 89/100
    1/1 - 0s - loss: 0.2376 - accuracy: 0.5000 - val_loss: 0.2374 - val_accuracy: 0.5000
    Epoch 90/100
    1/1 - 0s - loss: 0.2374 - accuracy: 0.5000 - val_loss: 0.2373 - val_accuracy: 0.5000
    Epoch 91/100
    1/1 - 0s - loss: 0.2373 - accuracy: 0.5000 - val_loss: 0.2371 - val_accuracy: 0.5000
    Epoch 92/100
    1/1 - 0s - loss: 0.2371 - accuracy: 0.5000 - val_loss: 0.2369 - val_accuracy: 0.5000
    Epoch 93/100
    1/1 - 0s - loss: 0.2369 - accuracy: 0.5000 - val_loss: 0.2367 - val_accuracy: 0.5000
    Epoch 94/100
    1/1 - 0s - loss: 0.2367 - accuracy: 0.5000 - val_loss: 0.2365 - val_accuracy: 0.5000
    Epoch 95/100
    1/1 - 0s - loss: 0.2365 - accuracy: 0.5000 - val_loss: 0.2363 - val_accuracy: 0.5000
    Epoch 96/100
    1/1 - 0s - loss: 0.2363 - accuracy: 0.5000 - val_loss: 0.2361 - val_accuracy: 0.5000
    Epoch 97/100
    1/1 - 0s - loss: 0.2361 - accuracy: 0.5000 - val_loss: 0.2359 - val_accuracy: 0.5000
    Epoch 98/100
    1/1 - 0s - loss: 0.2359 - accuracy: 0.5000 - val_loss: 0.2357 - val_accuracy: 0.5000
    Epoch 99/100
    1/1 - 0s - loss: 0.2357 - accuracy: 0.5000 - val_loss: 0.2355 - val_accuracy: 0.5000
    Epoch 100/100
    1/1 - 0s - loss: 0.2355 - accuracy: 0.5000 - val_loss: 0.2354 - val_accuracy: 0.5000
    1/1 - 0s - loss: 0.2354 - accuracy: 0.5000
    




    [0.2353529930114746, 0.5]




```python
plt.figure(figsize=(8, 4)) # 8 x 4 inchs
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("Cost Graph")
plt.ylabel("cost")
plt.subplot(1, 2, 2)
plt.title("Performance Graph")
plt.ylabel("performance")
plt.plot(hist.history['accuracy'], 'b-', label="training accuracy")
plt.plot(hist.history['val_accuracy'], 'r:', label="validation accuracy")
plt.legend()
plt.tight_layout()
plt.show()
print()
```


    
![output_62_0](https://user-images.githubusercontent.com/86539195/135233823-91e4ade3-cdad-41bd-a65c-da45fa08ba31.png)
    


    
    


```python

```
