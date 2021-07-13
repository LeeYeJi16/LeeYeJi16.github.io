---
layout: post
title:  "데이터 분석 심화 2 - 군집 모델"
categories: ['programming','python','data analysis']
---


# 데이터 분석 심화 2 - 군집 모델
## A. 클러스터링이 무엇인가
1. 클러스터링의 개념
- 데이터 개체 집합을 하위 집합으로 분할하는 과정
- 하위 집합을 클러스터라고 함
- 예측보다는 지식 추출에 사용

2. 차이점
- 분류: KNN, Decision Tree
- 회귀: Linear and Logistic regression
- 클러스터링: K-Means, Agglomerative Filtering, DBSCAN

3. 적용
- 타겟 마케팅 캠페인을 위해 유사한 인구통계나 구매 패턴을 가진 그룹으로 고객 세분화
- 비정상 행동 탐지: 알려진 클러스터를 벗어나는 사용 패턴을 식별하여 무단 네트워크 침입
- 매우 큰 데이터셋 단순화: 비슷한 값을 가진 특징을 더 적은 수의 동종 범주로 그룹화

## B. 클러스터링 기술
### K-Means
- K는 클러스터의 수
- 중심 기반 기술: 중심은 각 클러스터에 속한 개체의 평균
- 가장 가까운 중심으로 각 개체를 그룹화

### Python에서의 K-Means


```python
from sklearn.cluster import KMeans
import numpy as np
```


```python
X = np.array([[2, 10], [2, 5], [8, 4],[5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
X
```




    array([[ 2, 10],
           [ 2,  5],
           [ 8,  4],
           [ 5,  8],
           [ 7,  5],
           [ 6,  4],
           [ 1,  2],
           [ 4,  9]])




```python
n_clusters=3
```


```python
kmeans = KMeans(n_clusters)
```


```python
kmeans.fit(X)
```




    KMeans(n_clusters=3)




```python
# kmeans = KMeans(n_clusters=3).fit(X)

print("Labels: ", kmeans.labels_)
print("Cluster Centers: ", kmeans.cluster_centers_)
print("Predict Values: ", kmeans.predict([[1, 1]]))
```

    Labels:  [0 2 1 0 1 1 2 0]
    Cluster Centers:  [[3.66666667 9.        ]
     [7.         4.33333333]
     [1.5        3.5       ]]
    Predict Values:  [2]
    

### Python에서의 K-Means 시각화


```python
import matplotlib.pyplot as plt
```


```python
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black',marker="+", s=200)
plt.show()
```


    
![output_10_0](https://user-images.githubusercontent.com/86539195/125451899-578ebd31-2aa9-47b6-bba5-024f001d897e.png)
    



```python
# 실습해보기

X = np.array([[2, 10], [2, 5], [8, 4],[5, 8], [7, 5], [6, 4], [1, 2], [4, 9],[3,9],[8,2],[6,4],[7,2],[1,6],[4,4],[6,3],[2,7],[1,1]])
kmeans = KMeans(n_clusters=4).fit(X)

print("Labels: ", kmeans.labels_)
print("Cluster Centers: ", kmeans.cluster_centers_)
print("Predict Values: ", kmeans.predict([[1, 1]]))

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black',marker="+", s=200)
plt.show()
```

    Labels:  [2 3 1 2 1 1 0 2 2 1 1 1 3 3 1 3 0]
    Cluster Centers:  [[1.         1.5       ]
     [6.85714286 3.42857143]
     [3.5        9.        ]
     [2.25       5.5       ]]
    Predict Values:  [0]
    


    
![output_11_1](https://user-images.githubusercontent.com/86539195/125451891-795338f7-7ccb-4d1d-8d54-052185438dde.png)
    


#### ● r15 dataset


```python
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

sample_df = pd.read_csv("r15.csv")
sample_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.802</td>
      <td>10.132</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.350</td>
      <td>9.768</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.098</td>
      <td>9.988</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.730</td>
      <td>9.910</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.754</td>
      <td>10.430</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>14.198</td>
      <td>4.804</td>
      <td>15</td>
    </tr>
    <tr>
      <th>596</th>
      <td>14.320</td>
      <td>4.590</td>
      <td>15</td>
    </tr>
    <tr>
      <th>597</th>
      <td>13.636</td>
      <td>5.218</td>
      <td>15</td>
    </tr>
    <tr>
      <th>598</th>
      <td>14.410</td>
      <td>4.656</td>
      <td>15</td>
    </tr>
    <tr>
      <th>599</th>
      <td>14.020</td>
      <td>5.614</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 3 columns</p>
</div>




```python
# 원데이터 확인
training_points = sample_df[["col1", "col2"]]
training_labels = sample_df["target"]

plt.scatter(training_points["col1"], training_points["col2"], c=training_labels,cmap='rainbow')

plt.show()
```


    
![output_14_0](https://user-images.githubusercontent.com/86539195/125451876-3e22fc1f-193e-4269-8738-03bb7278d133.png)
    



```python
n_clusters=15
```


```python
# 모델 생성
kmeans = KMeans(n_clusters)

# 학습
kmeans.fit(training_points)
```




    KMeans(n_clusters=15)




```python
# 시각화
plt.scatter(training_points["col1"], training_points["col2"], c=kmeans.labels_,cmap='rainbow')

plt.show()
```


    
![output_17_0](https://user-images.githubusercontent.com/86539195/125451854-793057c5-75f7-4dd3-b86f-ac2a44fd1577.png)
    


#### ● spiral dataset


```python
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# 원데이터 확인
sample_df = pd.read_csv("spiral.csv")

training_points = sample_df[["X", "Y"]]
training_labels = sample_df["Group"]

plt.scatter(training_points["X"], training_points["Y"], c=training_labels,cmap='rainbow')

plt.show()
```


    
![output_20_0](https://user-images.githubusercontent.com/86539195/125451837-c1e8576e-9df3-456e-a183-349de7151ba9.png)
    



```python
# 모델 생성 / 시각화
kmeans = KMeans(n_clusters=3).fit(training_points)

plt.scatter(training_points["X"], training_points["Y"], c=kmeans.labels_,cmap='rainbow')

plt.show()

# 분류가 잘 되기 힘듦
```


    
![output_21_0](https://user-images.githubusercontent.com/86539195/125451821-94c9a388-623a-48da-a0c8-8768bdc3ac23.png)
    


### Python에서의 병합 클러스터링


```python
from sklearn.cluster import AgglomerativeClustering
```

#### ● r15 dataset


```python
sample_df = pd.read_csv("r15.csv")
training_points = sample_df[["col1", "col2"]]
training_labels = sample_df["target"]

# agglo = AgglomerativeClustering(n_clusters=15)
# agglo.fit(training_points)
agglo = AgglomerativeClustering(n_clusters=15).fit(training_points)

plt.scatter(training_points["col1"], training_points["col2"], c=agglo.labels_,cmap='rainbow')

plt.show()
```


    
![output_25_0](https://user-images.githubusercontent.com/86539195/125451802-bbf17cae-10bc-4471-bcee-6a3b1f0802d0.png)
    


#### ● spiral dataset


```python
sample_df = pd.read_csv("spiral.csv")
training_points = sample_df[["X", "Y"]]
training_labels = sample_df["Group"]

agglo = AgglomerativeClustering(n_clusters=3).fit(training_points)

plt.scatter(training_points["X"], training_points["Y"], c=agglo.labels_,cmap='rainbow')

plt.show()

# 병합 클러스터링을 써도 분류가 잘 나오기 힘듦
```


    
![output_27_0](https://user-images.githubusercontent.com/86539195/125451785-a99435fb-03f1-4d23-b1f8-b973f7e8decb.png)
    


### DBSCAN
- 연결된 점을 기반으로 하는 클러스터링
- “이웃”의 밀도가 일부 임계값을 초과하는 한 주어진 클러스터를 계속 확장

### Python에서의 DBSCAN


```python
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
```

#### ● r15 dataset


```python
sample_df = pd.read_csv("r15.csv")
training_points = sample_df[["col1", "col2"]]
training_labels = sample_df["target"]

# EPS=0.6
# MIN_SAMPLES=10
# dbscan = DBSCAN(eps=EPS, min_samples=MIN_SMAPLES)
# dbscan.fit(training_points)
dbscan = DBSCAN(eps=0.6, min_samples=10).fit(training_points)

plt.scatter(training_points["col1"], training_points["col2"], c=dbscan.labels_,cmap='rainbow')

plt.show()

# (-)값을 가장 줄여주는 eps 찾기
```


    
![output_32_0](https://user-images.githubusercontent.com/86539195/125451753-dbef5c98-91f7-4f05-9964-d39afefc6514.png)
    


#### ● spiral dataset


```python
sample_df = pd.read_csv("spiral.csv")
training_points = sample_df[["X", "Y"]]
training_labels = sample_df["Group"]

dbscan = DBSCAN(eps=2, min_samples=1).fit(training_points)

plt.scatter(training_points["X"], training_points["Y"], c=dbscan.labels_,cmap='rainbow')

plt.show()
```


    
![output_34_0](https://user-images.githubusercontent.com/86539195/125451731-169feff0-2828-4016-b34a-988f8f5467b9.png)
    


## C. 클러스터링 평가

- 조정 랜드 지수를 이용해 모델을 평가할 수 있다.
- 지수가 클수록 모델이 좋다고 할 수 있다.

### K-Means를 위한 조정 랜드 지수


```python
from sklearn.metrics.cluster import adjusted_rand_score
```

#### ● r15 dataset


```python
sample_df = pd.read_csv("r15.csv")
training_points = sample_df[["col1", "col2"]]
training_labels = sample_df["target"]

kmeans = KMeans(n_clusters=15).fit(training_points)

arc = adjusted_rand_score(training_labels, kmeans.labels_)
print(arc)
```

    0.9927781994136302
    

#### ● spiral dataset


```python
sample_df = pd.read_csv("spiral.csv")
training_points = sample_df[["X", "Y"]]
training_labels = sample_df["Group"]

dbscan = DBSCAN(eps=2, min_samples=1).fit(training_points)

arc = adjusted_rand_score(training_labels, dbscan.labels_)
print(arc)
```

    1.0
    


```python

```
