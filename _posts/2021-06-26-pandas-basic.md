---
layout: post
title:  "파이썬 - numpy 실습"
categories: ['programming','python']
---

# 파이썬 - pandas 기초 실습

## 1. pandas 개요

### pandas 특징

- 빅데이터 분석에 최적화 된 필수 패키지
- 데이터는 시계열(series)이나 표(table)의 형태
- 표 데이터를 다루기 위한 시리즈(series) 클래스 변환
- 데이터프레임(dataframe) 클래스 변환


```python
#pandas 패키지 import
import pandas as pd
import numpy as np
```

## 2. 시리즈 클래스 (series class)

### 2.1 시리즈 정의

- 데이터를 리스트나 1차원 배열 형식으로 series 클래스 생성자에 넣어주면 시리즈 클래스 객체를 만들 수 있다.
- 시리즈 클래스는 NumPy에서 제공하는 1차원 배열과 비슷하지만 각 데이터의 의미를 표시하는 인덱스(index)를 붙일 수 있다. 데이터 자체는 값(value)라고 한다.

 시리즈 = 인덱스(index) + 값(value) 


```python
# Series 정의하여 생성하기
obj = pd.Series([4, 5, -2, 8]) 
obj 
```




    0    4
    1    5
    2   -2
    3    8
    dtype: int64



### 2.2 시리즈 확인


```python
# Series의 값만 확인하기 
obj.values
```




    array([ 4,  5, -2,  8], dtype=int64)




```python
# series의 인덱스 확인하기
obj.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
# series의 데이터타입 확인하기
obj.dtypes
```




    dtype('int64')



### 2.3 시리즈의 인덱스

- 인덱스의 길이는 데이터의 길이와 같아야 하며, 인덱스의 값은 인덱스 라벨(label)
- 인덱스 라벨은 문자열 뿐 아니라 날짜, 시간, 정수 등도 가능


```python
# 인덱스를 리스트로 별도 지정, 반드시 " " 쌍따옴표 사용
obj1 = pd.Series([4, 5, -2, 8], index=["a", "b", "c", "d"]) 
obj1
```




    a    4
    b    5
    c   -2
    d    8
    dtype: int64



### 2.4 시리즈와 딕셔너리 자료형

- Python의 dictionary 자료형을 Series data로 만들 수 있다.
- dictionary의 key가 Series의 index가 된다


```python
data = {"Kim": 35000, "Park": 67000, "Joon": 12000, "Choi": 4000}
```


```python
obj2 = pd.Series(data) 
obj2
```




    Kim     35000
    Park    67000
    Joon    12000
    Choi     4000
    dtype: int64




```python
# 시리즈 이름 지정 및 index name 지정  
obj2.name = "Salary" 
obj2.index.name = "Names" 

obj2
```




    Names
    Kim     35000
    Park    67000
    Joon    12000
    Choi     4000
    Name: Salary, dtype: int64




```python
# index 이름 변경 
obj2.index = ["A", "B", "C", "D"] 
obj2 
```




    A    35000
    B    67000
    C    12000
    D     4000
    Name: Salary, dtype: int64



### 2.5 시리즈 연산

- Numpy 배열처럼 시리즈도 벡터화 연산 가능
- 시리즈의 값에만 적용되며 인덱스 값은 변하지 않는다


```python
obj * 10
```




    0    40
    1    50
    2   -20
    3    80
    dtype: int64




```python
# 인덱싱 끼리 연산
obj1 * obj1
```




    a    16
    b    25
    c     4
    d    64
    dtype: int64




```python
# values 값끼리 연산
obj1.values + obj.values
```




    array([ 8, 10, -4, 16], dtype=int64)



### 2.6 시리즈 인덱싱

- 시리즈는 numpy 배열의 인덱스 방법처럼 사용 외에 인덱스 라벨을 이용한 인덱싱
- 배열 인덱싱은 자료의 순서를 바꾸거나 특정한 자료만 선택 가능
- 라벨 값이 영문 문자열인 경우에는 마치 속성인것처럼 점(.)을 이용하여 접근


```python
a = pd.Series([1024, 2048, 3096, 6192],
              index=["서울", "부산", "인천", "대구"])
a
```




    서울    1024
    부산    2048
    인천    3096
    대구    6192
    dtype: int64




```python
a[1], a["부산"]
```




    (2048, 2048)




```python
a[3], a["대구"]
```




    (6192, 6192)




```python
a[[0, 3, 1]]
```




    서울    1024
    대구    6192
    부산    2048
    dtype: int64




```python
a[["서울", "대구", "부산"]]
```




    서울    1024
    대구    6192
    부산    2048
    dtype: int64




```python
# 라벨 값이 영문 문자열인 경우에는 마치 속성인것처럼 점(.)을 이용하여 접근
obj2.A
```




    35000




```python
obj2.C
```




    12000



### 2.7 시리즈 슬라이싱(slicing)

- 배열 인덱싱이나 인덱스 라벨을 이용한 슬라이싱(slicing)도 가능
- 문자열 라벨을 이용한 슬라이싱은 콜론(:) 기호 뒤에 오는 인덱스에 해당하는 값이 결과에 포함


```python
a[1:3]
```




    부산    2048
    인천    3096
    dtype: int64




```python
a["부산":"대구"]
```




    부산    2048
    인천    3096
    대구    6192
    dtype: int64



### 2.8 시리즈의 데이터 갱신, 추가, 삭제

- 인덱싱을 이용하여 딕셔너리처럼 데이터를 갱신(update)하거나 추가(add)
- 데이터 삭제 시 딕셔너리처럼 del 명령 사용


```python
# 데이터 갱신
a["부산"] = 1234
a
```




    서울    1024
    부산    1234
    인천    3096
    대구    6192
    dtype: int64




```python
# 데이터 갱신
a["대구"] = 6543
a
```




    서울    1024
    부산    1234
    인천    3096
    대구    6543
    dtype: int64




```python
# del 명령어로 데이터 삭제
del a["서울"]
a
```




    부산    1234
    인천    3096
    대구    6543
    dtype: int64



## 3. 데이터프레임(DataFrame)
### 3.1 데이터프레임(DataFrame) 개요

- 시리즈가 1차원 벡터 데이터에 행 방향 인덱스(row index)이라면,
- 데이터프레임(data-frame) 클래스는 2차원 행렬 데이터에 합친 것으로
- 행 인덱스(row index)와 열 인덱스(column index)를 지정

 데이터프레임 = 시리즈{인덱스(index) + 값(value)} + 시리즈 + 시리즈의 연속체

### 3.2 데이터프레임 특성

- 데이터프레임은 공통 인덱스를 가지는 열 시리즈(column series)를 딕셔너리로 묶어놓은 것
- 데이터프레임은 numpy의 모든 2차원 배열 속성이나 메서드를 지원

### 3.3 데이터프레임 생성

1. 우선 하나의 열이 되는 데이터를 리스트나 일차원 배열을 준비
2. 각 열에 대한 이름(label)의 키(key)를 갖는 딕셔너리를 생성
3. pandas의 DataFrame 클래스로 생성
4. 열방향 인덱스는 columns 인수로, 행방향 인덱스는 index 인수로 지정


```python
# Data Frame은 python의 dictionary 또는 numpy의 array로 정의
data = {
'name': ["Choi", "Choi", "Choi", "Kim", "Park"], 
'year': [2013, 2014, 2015, 2016, 2017], 
'points': [1.5, 1.7, 3.6, 2.4, 2.9]
} 
df = pd.DataFrame(data) 
df 
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
      <th>name</th>
      <th>year</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Choi</td>
      <td>2013</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Choi</td>
      <td>2014</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Choi</td>
      <td>2015</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kim</td>
      <td>2016</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Park</td>
      <td>2017</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행 방향의 index 
df.index
```




    RangeIndex(start=0, stop=5, step=1)




```python
# 열 방향의 index 
df.columns
```




    Index(['name', 'year', 'points'], dtype='object')




```python
# 값 얻기
df.values
```




    array([['Choi', 2013, 1.5],
           ['Choi', 2014, 1.7],
           ['Choi', 2015, 3.6],
           ['Kim', 2016, 2.4],
           ['Park', 2017, 2.9]], dtype=object)



### 3.4 데이터프레임 열 갱신 추가

- 데이터프레임은 열 시리즈의 딕셔너리로 볼 수 있으므로 열 단위로 데이터를 갱신하거나 추가, 삭제
- data에 포함되어 있지 않은 값은 nan(not a number)으로 나타내는 null과 같은 개념
- 딕셔너리, numpy의 배열, 시리즈의 다양한 방법으로 추가 가능


```python
# DataFrame을 만들면서 columns와 index를 설정
df = pd.DataFrame(data, columns=["year", "name", "points", "penalty"],
                                  index=(["one", "two", "three", "four", "five"]))
df
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 특정 열만 선택
df[["year","points"]]
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
      <th>year</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 특정 열을 선택하고, 값(0.5)을 대입 
df["penalty"] = 0.5 
df
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 또는 python의 리스트로 대입
df['penalty'] = [0.1, 0.2, 0.3, 0.4, 0.5] 
df 
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 또는 numpy의 np.arange로 새로운 열을 추가하기
df['zeros'] = np.arange(5) 
df 
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>zeros</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 또는 index인자로 특정행을 지정하여 시리즈(Series)로 추가
val = pd.Series([-1.2, -1.5, -1.7], index=['two','four','five']) 

df['debt'] = val 
df
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>zeros</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>1</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>3</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>4</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 연산 후 새로운 열을 추가하기
df["net_points"] = df["points"] - df["penalty"]
df
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>zeros</th>
      <th>debt</th>
      <th>net_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>1</td>
      <td>-1.2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>2</td>
      <td>NaN</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>3</td>
      <td>-1.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>4</td>
      <td>-1.7</td>
      <td>2.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 조건 연산으로 열 추가
df["high_points"] = df["net_points"] > 2.0 
df
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>zeros</th>
      <th>debt</th>
      <th>net_points</th>
      <th>high_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.4</td>
      <td>False</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>1</td>
      <td>-1.2</td>
      <td>1.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>2</td>
      <td>NaN</td>
      <td>3.3</td>
      <td>True</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>3</td>
      <td>-1.5</td>
      <td>2.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>4</td>
      <td>-1.7</td>
      <td>2.4</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 열 삭제하기 
del df["high_points"] 
del df["net_points"] 
del df["zeros"] 
df
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
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 컬럼명 확인하기
df.columns
```




    Index(['year', 'name', 'points', 'penalty', 'debt'], dtype='object')




```python
# index와 columns 이름 지정
df.index.name = "Order" 
df.columns.name = "Info" 
df
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>



### 3.4 데이터프레임 인덱싱

#### 열 인덱싱
- 데이터프레임을 인덱싱을 할 때도 열 라벨(column label)을 키 값으로 생각하여 인덱싱
- 인덱스로 라벨 값을 하나만 넣으면 시리즈 객체가 반환되고 라벨의 배열 또는 리스트를 넣으면 부분적인 데이터프레임이 반환
- 하나의 열만 빼내면서 데이터프레임 자료형을 유지하고 싶다면 원소가 하나인 리스트를 써서 인덱싱

#### 행 인덱싱
- 행 단위로 인덱싱을 하고자 하면 항상 슬라이싱(slicing)을 해야 한다.
- 인덱스의 값이 문자 라벨이면 라벨 슬라이싱


```python
# 열 인덱싱
df["year"] 
```




    Order
    one      2013
    two      2014
    three    2015
    four     2016
    five     2017
    Name: year, dtype: int64




```python
# 다른 방법의 열 인덱싱
df.year
```




    Order
    one      2013
    two      2014
    three    2015
    four     2016
    five     2017
    Name: year, dtype: int64




```python
# 행 인덱싱은 슬라이싱으로 0번째부터 1번째로 지정하면 1행을 반환
df[0:1]
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행 인덱싱 슬라이싱으로 0번째 부터 2(3-1) 번째까지 반환
df[0:3] 
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 3.5 loc 인덱싱
인덱스의 라벨값 기반의 2차원 (행, 열)인덱싱


```python
# .loc 함수를 사용하여 시리즈로 인덱싱
df.loc["two"]
```




    Info
    year       2014
    name       Choi
    points      1.7
    penalty     0.2
    debt       -1.2
    Name: two, dtype: object




```python
# .loc 또는 .iloc 함수를 사용하여 데이터프레임으로 인덱싱
df.loc["two":"four"]
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>-1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc["two":"four", "points"] 
```




    Order
    two      1.7
    three    3.6
    four     2.4
    Name: points, dtype: float64




```python
# == df['year’] 
df.loc[:,'year']
```




    Order
    one      2013
    two      2014
    three    2015
    four     2016
    five     2017
    Name: year, dtype: int64




```python
df.loc[:,['year','name']] 
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013</td>
      <td>Choi</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014</td>
      <td>Choi</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc["three":"five","year":"penalty"] 
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>2015</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



### 3.6 iloc 인덱싱
인덱스의 숫자 기반의 2차원 (행, 열)인덱싱


```python
# 새로운 행 삽입하기 
df.loc['six',:] = [2013,'Jun',4.0,0.1,2.1] 
df
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013.0</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014.0</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015.0</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016.0</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017.0</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>-1.7</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2013.0</td>
      <td>Jun</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>2.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 4번째 행을 가져오기 위해 .iloc 사용:: index 번호를 사용
df.iloc[3] #3번째 행을 가져온다.
```




    Info
    year       2016.0
    name          Kim
    points        2.4
    penalty       0.4
    debt         -1.5
    Name: four, dtype: object




```python
# 슬라이싱으로 지정하여 반환
df.iloc[3:5, 0:2]
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>four</th>
      <td>2016.0</td>
      <td>Kim</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017.0</td>
      <td>Park</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 각각의 행과 열을 지정하여 반환하기
df.iloc[[0,1,3], [1,2]]
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
      <th>Info</th>
      <th>name</th>
      <th>points</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>Choi</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>two</th>
      <td>Choi</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>four</th>
      <td>Kim</td>
      <td>2.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행을 전체, 열은 두번째열부터 마지막까지 슬라이싱으로 지정하여 반환
df.iloc[:,1:4] 
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
      <th>Info</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>four</th>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>five</th>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>six</th>
      <td>Jun</td>
      <td>4.0</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행 두번째 열 두번째 
df.iloc[1,1]
```




    'Choi'



### 3.7 Boolean 인덱싱

True, False 논리연산 기반의 인덱싱


```python
df
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2013.0</td>
      <td>Choi</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2014.0</td>
      <td>Choi</td>
      <td>1.7</td>
      <td>0.2</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2015.0</td>
      <td>Choi</td>
      <td>3.6</td>
      <td>0.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2016.0</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017.0</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>-1.7</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2013.0</td>
      <td>Jun</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>2.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# year가 2014보다 큰 boolean data 
df["year"] > 2014 
```




    Order
    one      False
    two      False
    three     True
    four      True
    five      True
    six      False
    Name: year, dtype: bool




```python
df.loc[df['name'] == "Choi", ['name','points']]
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
      <th>Info</th>
      <th>name</th>
      <th>points</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>Choi</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>two</th>
      <td>Choi</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>three</th>
      <td>Choi</td>
      <td>3.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# numpy에서와 같이 논리연산을 응용할 수 있다. 
df.loc[(df["points"] > 2) & (df["points"]<3), :]
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
      <th>Info</th>
      <th>year</th>
      <th>name</th>
      <th>points</th>
      <th>penalty</th>
      <th>debt</th>
    </tr>
    <tr>
      <th>Order</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>four</th>
      <td>2016.0</td>
      <td>Kim</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2017.0</td>
      <td>Park</td>
      <td>2.9</td>
      <td>0.5</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>



## 4. 데이터프레임 다루기
### 4.1 numpy randn 데이터프레임 생성


```python
# DataFrame을 만들때 index, column을 설정하지 않으면 기본값으로 0부터 시작하는 정수형 숫자로 입력된다. 
df = pd.DataFrame(np.random.randn(6,4)) 
df 
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
    </tr>
  </tbody>
</table>
</div>



### 4.2 시계열 데이트 함수 date_range


```python
df.columns = ["A", "B", "C", "D"] 
```


```python
#pandas에서 제공하는 date range함수는 datetime 자료형으로 구성된, 날짜/시간 함수 
df.index = pd.date_range('20160701', periods=6)
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    DatetimeIndex(['2016-07-01', '2016-07-02', '2016-07-03', '2016-07-04',
                   '2016-07-05', '2016-07-06'],
                  dtype='datetime64[ns]', freq='D')



### 4.3 numpy로 데이터프레임 결측치 다루기


```python
# np.nan은 NaN값을 의미
df["F"] = [1.0, np.nan, 3.5, 6.1, np.nan, 7.0] 
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행의 값중 하나라도 nan인 경우 그 행을 없앤다. 
df.dropna(how="any") 
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 행의 값의 모든 값이 nan인 경우 그 행을 없앤다.
df.dropna(how='all')
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# NaN에 특정 value 값 넣기
df.fillna(value=0.5)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



### 4.4 drop 명령어


```python
# 특정 행 drop하기 
df.drop(pd.to_datetime('20160701'))
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 2개 이상도 가능 
df.drop([pd.to_datetime('20160702'),pd.to_datetime('20160704')])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 특정 열 삭제하기 
df.drop('F', axis = 1)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>-0.473855</td>
      <td>0.512795</td>
      <td>0.750526</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.928088</td>
      <td>0.406070</td>
      <td>0.350141</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>1.102393</td>
      <td>-0.587363</td>
      <td>0.788561</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>-1.403983</td>
      <td>0.108649</td>
      <td>-0.104171</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.083137</td>
      <td>0.917482</td>
      <td>0.501891</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>-2.300502</td>
      <td>1.088832</td>
      <td>1.793931</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 2개 이상의 열도 가능 
df.drop(['B','D'], axis = 1) 
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
      <th>A</th>
      <th>C</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>-0.302163</td>
      <td>0.512795</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-07-02</th>
      <td>1.900080</td>
      <td>0.406070</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-03</th>
      <td>-0.748171</td>
      <td>-0.587363</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>-1.451959</td>
      <td>0.108649</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.940441</td>
      <td>0.917482</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>0.419030</td>
      <td>1.088832</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



## 5. pandas 데이터 입출력
- pandas는 데이터 분석을 위해 여러 포맷의 데이터 파일을 읽고 쓸수 있다.
- csv, excel, html, json, hdf5, sas, stata, sql

### 5.1 pandas 데이터 불러오기


```python
pd.read_csv('data/sample1.csv')
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
      <th>c1</th>
      <th>c2</th>
      <th>c3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.11</td>
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2.22</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3.33</td>
      <td>three</td>
    </tr>
  </tbody>
</table>
</div>




```python
# c1을 인덱스로 불러오기
pd.read_csv('data/sample1.csv', index_col="c1")
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
      <th>c2</th>
      <th>c3</th>
    </tr>
    <tr>
      <th>c1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.11</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.22</td>
      <td>two</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.33</td>
      <td>three</td>
    </tr>
  </tbody>
</table>
</div>



### 5.2 pandas 데이터 쓰기


```python
#pandas는 데이터프레임을 출력하는데 여러가지 포맷을 지원
df.to_csv('sample6.csv')
```


```python
#index, header 인수를 지정하여 인덱스 및 헤더 출력 여부를 지정
df.to_csv('sample9.csv',index=False, header=False)
```

### 5.3 인터넷에서 데이터 불러오기
#### 인터넷 링크의 데이터 불러오기
df = pd.read_excel("링크")\
df = pd.read_csv("링크")

## 6. 데이터 처리하기
### 6.1 정렬(Sort)
데이터를 정렬로 sort_index는 인덱스 값을 기준으로, \
sort_values는 데이터 값을 기준으로 정렬


```python
# np.random으로 시리즈 생성
s = pd.Series(np.random.randint(6, size=100))
s.head()
```




    0    2
    1    0
    2    4
    3    5
    4    3
    dtype: int32




```python
# value_counts 메서드로 값을 카운트
s.value_counts()
```




    4    22
    3    21
    5    18
    2    14
    0    13
    1    12
    dtype: int64




```python
# sort_index 메서드로 정렬하기
s.value_counts().sort_index()
```




    0    13
    1    12
    2    14
    3    21
    4    22
    5    18
    dtype: int64




```python
# ascending=False 인자로 내림차순 정리
s.sort_values(ascending=False)
```




    50    5
    92    5
    63    5
    61    5
    59    5
         ..
    58    0
    96    0
    6     0
    64    0
    52    0
    Length: 100, dtype: int32



### 6.2 apply 함수
- 행이나 열 단위로 더 복잡한 처리를 하고 싶을 때는 apply 메서드를 사용
- 인수로 행 또는 열을 받는 함수를 apply 메서드의 인수로 넣으면 각 열(또는 행)을 반복하여 수행

#### lambda 함수:
- 파이썬에서 "lambda" 는 런타임에 생성해서 사용할 수 있는 익명 함수
- lambda는 쓰고 버리는 일시적인 함수로 생성된 곳에서만 적용


```python
df = pd.DataFrame({
    'A': [1, 3, 4, 3, 4],
    'B': [2, 3, 1, 2, 3],
    'C': [1, 5, 2, 4, 4]
})
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 람다 함수 사용
df.apply(lambda x: x.max() - x.min())
```




    A    3
    B    2
    C    4
    dtype: int64




```python
# 만약 행에 대해 적용하고 싶으면 axis=1 인수 사용
df.apply(lambda x: x.max() - x.min(), axis=1)
```




    0    1
    1    2
    2    3
    3    2
    4    1
    dtype: int64



nan 값은 fillna 메서드를 사용하여 원하는 값으로 변환 가능\
astype 메서드로 전체 데이터의 자료형을 바꾸는 것도 가능


```python
# apply로 value_counts로 값의 수를 반환
df.apply(pd.value_counts)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# NaN 결측치에 fillna(0)으로 0을 채우고 순차적으로 정수로 변환
df.apply(pd.value_counts).fillna(0).astype(int)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 6.3 describe 메서드
describe() 함수는 DataFrame의 계산 가능한 값들의 통계값을 보여준다


```python
df.describe()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.000000</td>
      <td>5.00000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.000000</td>
      <td>2.20000</td>
      <td>3.200000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.224745</td>
      <td>0.83666</td>
      <td>1.643168</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>2.00000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.00000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>3.00000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>3.00000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 7. pandas 시계열 분석
### 7.1 pd.to_datetime 함수
날짜/시간을 나타내는 문자열을 자동으로 datetime 자료형으로 바꾼 후 \
datetimeindex 자료형 인덱스를 생성


```python
date_str = ["2018, 1, 1", "2018, 1, 4", "2018, 1, 5", "2018, 1, 6"]

idx = pd.to_datetime(date_str)
idx
```




    DatetimeIndex(['2018-01-01', '2018-01-04', '2018-01-05', '2018-01-06'], dtype='datetime64[ns]', freq=None)




```python
# 인덱스를 사용하여 시리즈나 데이터프레임을 생성
np.random.seed(0)

s = pd.Series(np.random.randn(4), index=idx)
s
```




    2018-01-01    1.764052
    2018-01-04    0.400157
    2018-01-05    0.978738
    2018-01-06    2.240893
    dtype: float64



### 7.2 pd.date_range 함수
시작일과 종료일 또는 시작일과 기간을 입력하면 범위 내의 인덱스를 생성


```python
pd.date_range("2018-4-1", "2018-4-5")
```




    DatetimeIndex(['2018-04-01', '2018-04-02', '2018-04-03', '2018-04-04',
                   '2018-04-05'],
                  dtype='datetime64[ns]', freq='D')




```python
pd.date_range(start="2018-4-1", periods=5)
```




    DatetimeIndex(['2018-04-01', '2018-04-02', '2018-04-03', '2018-04-04',
                   '2018-04-05'],
                  dtype='datetime64[ns]', freq='D')




```python

```
