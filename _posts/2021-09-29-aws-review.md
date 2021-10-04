---
layout: posts
title: AWS X CJ대한통운 물류 예측 PoC 
categories: ['interest']
tags: [aws, automl]
---

추천, 개인화 그리고 물류 예측 - 어떻게 시작하고 무엇을 준비해야 하는가? - 김민성 솔루션즈 아키텍트(AWS), 경희정 부장(CJ대한통운)
=======================================

{% include video id="PtHVqCDsoek" provider="youtube" %}

> Amazon Forecast
> 
> CJ대한통운과 AWS의 PoC
> 
> AWS의 AutoML

Amazon Forecast
----------------
![1 (1)](https://user-images.githubusercontent.com/86539195/135284174-fc970d4f-d48b-43b0-8be6-bde76337fcb9.png)

Amazon은 다수의 물류센터 보유, 제품도 판매(ex.Alexa), 로지스틱스도 운영하고 있음   
AWS는 Forecast Service를 제공

![1 (2)](https://user-images.githubusercontent.com/86539195/135283294-1f3d4fe0-6fec-460b-b38e-9d88fa310fa9.png)

forecast를 사용하기 위해서는 위와 같은 데이터 준비가 필요하다.

CJ대한통운과 AWS의 PoC
----------------
![1 (3)](https://user-images.githubusercontent.com/86539195/135283298-1d4fd673-3f13-4dd4-818a-bf4086060500.png)

![1 (4)](https://user-images.githubusercontent.com/86539195/135283301-4d6b58ae-6870-41b6-92ce-9f2ac56abdd5.png)

![1 (5)](https://user-images.githubusercontent.com/86539195/135283303-3266c616-340a-41ad-a845-458bf9c9d664.png)

빅데이터 조직은 Engineering조직에 속하며   
수배송 네트워크 및 물류센터 최적화를 위해 대량의 데이터를 분석하고   
사업의 value chain 확장을 위해 신사업 개발 및 서비스 차별화를 하고있다.

![1 (6)](https://user-images.githubusercontent.com/86539195/135283306-15167df4-0e7b-42f2-a623-85b0fcf6c7e1.png)

![1 (7)](https://user-images.githubusercontent.com/86539195/135283310-d9f3e86b-a596-4854-870e-e57d55035faf.png)

기존 CJ대한통운의 예측 모형은 서브터미널별로 적용되고 있고   
알고리즘은 위와 같은 알고리즘을 사용하였다.

![1 (8)](https://user-images.githubusercontent.com/86539195/135283312-2c823a67-cb52-4e50-9c0c-0b4fcc0c6cd8.png)

![1 (9)](https://user-images.githubusercontent.com/86539195/135283315-a39e7c7f-3af3-4284-b19d-ccf863e4ca3d.png)

AWS Forecast PoC 결과 DeepAR+보다 Forecast Beta 서비스가 더 예측 정확도가 낮았다.   
이러한 결과가 나온 이유는 서비스에 한국 캘린더 정보가 아직 추가되지 않아서 휴일 변수가 사용되지 못했기 때문이다.

AWS의 AutoML
------------
![1 (10)](https://user-images.githubusercontent.com/86539195/135283317-6acb1a71-a895-4a4e-ae1d-6e47b4100bb2.png)

AutoML: Forecast 안에는 11개의 알고리즘이 속해있고 이중 최적의 알고리즘을 자동으로 선택해준다.   
단점은 알고리즘이 어떻게 실행되는지(내 생각으로는 사용되는 하이퍼파라미터들이 무엇인지를 말하는 것같다.)   
어떤 변수가 영향을 많이주는지 알 수 없다는 점이다.   
또 한정적인 변수 사용이 아쉽다고 하였다.

![1 (11)](https://user-images.githubusercontent.com/86539195/135283321-2db24ab2-4acf-49dc-a010-5d97bda086d4.png)

![1 (12)](https://user-images.githubusercontent.com/86539195/135283326-d0e96698-0fd4-4273-b5e3-28645c3747c8.png)

내가 사용해본 알고리즘은 ARIMA, ETS, Prophet 정도   
나머지는 알아보자!

https://docs.aws.amazon.com/ko_kr/forecast/latest/dg/aws-forecast-choosing-recipes.html
