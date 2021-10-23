---
layout: posts
title: LightGBM 파라미터 튜닝하기
categories: ['algorithm']
tags: [LightGBM, parameter]
---

LightGBM parameter 
==========

num_leaves
---------
maximum leaves. 각 트리의 최대 리프 수. 트리 최대 깊이, 모형 성능, 학습 속도에 영향. 높은 정확도를 위해 크게하면 과적합이 될 수 있음. num_leaves = 2^(max_depth) 보다 작게 설정해야 과적합을 줄일 수 있음 (max_depth가 7일 경우 좋은 성능을 보였다면, num_leaves는 127보다 적은 70~80사이로). default = 31.

learning_rate
-------
부스팅 각 이터레이션마다 곱해지는 가중치. 모형성능, 힉습속도에 영향. 높은 정확도를 위해 num_iterations 크게, learning_rate 작게. default = 0.1

n_estimators
----------
부스팅 이터레이션 수. num_iteration 과 같음. 클수록 과적합 됨. default = 100. 1000정도 해주고 early_stopping_rounds = 50 으로 과적합 방지. 

max_bin
------
feature 값의 최대 bin 수. 예측력에 영향. default = 255. 크게하면 정확도 높으나 과적합 우려, 디폴트로 놔두는게 좋음.

bagging_fraction
------
각 이터레이션에 사용되는 행의 비율. default = 1. 0-1사이 값으로 설정. 민감한 옵션이므로, Column sampling과 잘 섞어서 쓴다. bagging_freq = 0 인데 1로 하면 XGBoost와 동일한 방식으로 학습.

feature_fraction
-------
각 이터레이션에서 사용되는 칼럼의 비율. default = 1. 보통 0.7~0.9 사용. 0.8 = 매 트리를 구성할 때 feature의 80%만 랜덤하게 선택한다는 것.

min_data_in_leaf
----------
리프가 가지고있는 최소한의 행 수. default = 20이고 최적값. 과적합을 해결하기 위해 사용.
 
min_sum_hessian_in_leaf
---------
과적합을 줄이기 위해 min_data_in_leaf와 같이 사용.   


   


회귀모델에 적용하기
==========
