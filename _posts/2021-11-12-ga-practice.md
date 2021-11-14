---
layout: posts
title: Google Analytics 실습하기
categories: ['interest']
tags: [GA, BA, PM]
---

1.필터
====
![screencapture-analytics-google-analytics-web-2021-11-12-15_23_58](https://user-images.githubusercontent.com/86539195/141420331-ff434ec4-b206-473f-9d0d-b1631e26c1ff.png)

웹사이트 관리 시 본인의 데이터가 쌓이게 된다면 데이터가 오염될 수 있음   
사용자들의 이용 행태를 분석하는 것이 목적이므로 본인의 IP는 분석 대상에서 제외함   
*PC외 모바일 등의 다른 디바이스에서 접속할 경우도 내부유입이 발생할 수 있음*

![Inkedscreencapture-analytics-google-analytics-web-2021-11-12-15_13_49_LI](https://user-images.githubusercontent.com/86539195/141420184-58580542-0e6a-487b-bb3d-25343b5d4628.jpg)

사용자 수가 0이 된 것을 확인할 수 있음

![screencapture-analytics-google-analytics-web-2021-11-12-15_15_02](https://user-images.githubusercontent.com/86539195/141420079-12b15e90-4b30-45f8-90a9-0addb936c5f2.png)


2.목표
===
목표는 한번 만들면 수정할 수 없으므로 신중하게 만들어야 함   
테스트 보기를 이용하면 실수 없이 만들 수 있음   

![screencapture-analytics-google-analytics-web-2021-11-12-15_43_36](https://user-images.githubusercontent.com/86539195/141422444-d84d2ce2-69ad-4f4e-800c-07e4a2e2290a.png)

목표 설정
- 템플릿
- 맞춤설정 ◀

목표 유형
- 도착 ◀
- 시간
- 세션당 페이지 수
- 이벤트

목표 세부정보
- 도착페이지가 될 URL 붙여넣기

![screencapture-analytics-google-analytics-web-2021-11-12-15_44_09](https://user-images.githubusercontent.com/86539195/141422453-1f0577a9-cdbc-419c-8869-78e71f7bcc82.png)

목표는 실시간 보고서 > 전환수 보고서에서 확인할 수 있음

테스트보기를 전체 웹사이트 보고서에 옮기기 위해서는 공유 → 템플릿 링크 공유 → URL 복사해 새 탭에 붙여넣기 → 보기선택창 실행


3.GA 기본세팅
====

속성 > 추적정보
-----

속성 > 추적 정보 > 데이터 보관 > 사용자 및 이벤트 데이터 보관

![screencapture-analytics-google-analytics-web-2021-11-14-10_35_29](https://user-images.githubusercontent.com/86539195/141664331-1b6d5034-6dc4-4ae8-9de8-65ad6cb7be64.png)

'자동만료안함'으로 설정해 이전 데이터 소실되지 않게하기


속성 > 추적 정보 > 세션 설정

![screencapture-analytics-google-analytics-web-2021-11-14-10_49_07](https://user-images.githubusercontent.com/86539195/141664364-bc95df87-e2bc-46f0-aa7a-30f2705d8460.png)

세션만료시간을 설정할 수 있음

PC보기 / 모바일보기
-----
보기 > 보기 설정 > 보기 복사
*보기 복사하는 이유는 필터를 혼용하여 쓰고 있을 때 일일이 설정하는 수고를 덜기 위해서*

![screencapture-analytics-google-analytics-web-2021-11-14-10_54_53](https://user-images.githubusercontent.com/86539195/141664466-55e6a2f8-f598-49d6-93a3-f07f2b79552c.png)

pc보기의 경우 필터에서 모바일제외, 태블릿제외   
모바일보기의 경우 필터에서 pc제외, 태블릿제외   
필터를 추가한다   

![screencapture-analytics-google-analytics-web-2021-11-14-10_57_45](https://user-images.githubusercontent.com/86539195/141664566-da083c1a-d830-40d4-95e6-068e7766c988.png)
![screencapture-analytics-google-analytics-web-2021-11-14-11_00_37](https://user-images.githubusercontent.com/86539195/141664582-8b91e4fc-b269-435c-b8dc-fe302fb99f3f.png)
![screencapture-analytics-google-analytics-web-2021-11-14-11_02_36](https://user-images.githubusercontent.com/86539195/141664611-23a492c5-75f0-4a8a-b871-fdab264f4a86.png)

위와 같은 설정을 통해 pc와 모바일의 사용자를 각각 체킹할 수 있음 


4.세그먼트와 맞춤보고서
====

세그먼트
----

획득 > 전체 트래픽 > 소스/매체 > +세그먼트 버튼 클릭 > +새 세그먼트 버튼 클릭   

> 2월달에 처음 방문하였으면서, 구매를 한 번이라도 한 사람의 데이터 그룹 만들기

![화면 캡처 2021-11-14 111514](https://user-images.githubusercontent.com/86539195/141664899-d15d027e-5e6b-4f7b-9f90-afe0ece24745.png)

![화면 캡처 2021-11-14 111541](https://user-images.githubusercontent.com/86539195/141664902-4abf79ba-26fe-4788-a608-2f3b47b43924.png)

![화면 캡처 2021-11-14 112000](https://user-images.githubusercontent.com/86539195/141664904-3f56976a-c998-4e15-9362-1b50e8ebbf01.png)


맞춤보고서
-----

맞춤설정 > 맞춤보고서 > +새맞춤보고서 버튼 클릭   

> 목표보고서 만들기

![screencapture-analytics-google-analytics-web-2021-11-14-11_25_41](https://user-images.githubusercontent.com/86539195/141665002-c864d7f8-0076-4ceb-8e9d-79ebe10020b2.png)

![screencapture-analytics-google-analytics-web-2021-11-14-11_28_39](https://user-images.githubusercontent.com/86539195/141665067-2d0e9073-220b-440a-b064-ce11ed845847.png)

직접 URL 접속이 가장 큰 목표달성을 이룬 것을 확인할 수 있다   


※ 본 포스트는 인프런의 'Google Analytics(GA) 보고서 살펴보기' 강의를 참고하였습니다.
