---
layout: posts
title: "[Cloud Service] Data Catalog"
categories: ['cloud']
tags: [data catalog]
---

Data Catalog는 단순히 메타데이터를 보여주는 도구가 아니라, **데이터가 구조적으로 관리되고 재사용 가능한 상태인지 보장하는 핵심 구성 요소**다. SQA 관점에서 Data Catalog 테스트는 데이터 자체의 정확성보다는, **데이터 구조·변경·탐색 흐름이 안정적으로 유지되는지**를 확인하는 데 목적이 있다.

아래는 클라우드 환경에서 Data Catalog를 사용할 때, 실제 테스트에서 자주 확인하는 **파티션 기반 테이블 관리**와 **크롤러 동작 검증**을 중심으로 정리한다.

---

## 1. Data Catalog 테이블과 파티션 개념

Data Catalog에서 테이블은 HDFS 또는 오브젝트 스토리지 상의 데이터를 논리적으로 표현한 단위다. 이때 파티션은 데이터를 특정 기준(날짜, 지역, 타입 등)으로 나누어 관리하기 위한 구조다.

파티션을 사용하는 이유는 다음과 같다.

- 대용량 데이터 탐색 성능 개선
- 데이터 적재·변경 관리 용이
- 분석 쿼리 범위 제한

SQA 테스트에서는 **파티션이 의도한 기준으로 생성·인식되는지**가 핵심 확인 포인트다.

---

## 2. 테이블 내 파티션 생성 테스트

### ① 파티션 디렉터리 생성

예시로 날짜 기준 파티션을 사용하는 경우를 가정한다.

```bash
hdfs dfs -mkdir -p /data/events/dt=2025-01-01
hdfs dfs -mkdir -p /data/events/dt=2025-01-02
```

테스트용 데이터 파일을 각 파티션 디렉터리에 업로드한다.

```bash
hdfs dfs -put event_0101.csv /data/events/dt=2025-01-01/
hdfs dfs -put event_0102.csv /data/events/dt=2025-01-02/
```

---

### ② Data Catalog 테이블 생성 (외부 테이블 예시)

```sql
CREATE EXTERNAL TABLE events (
  user_id STRING,
  event_type STRING,
  ts STRING
)
PARTITIONED BY (dt STRING)
STORED AS TEXTFILE
LOCATION '/data/events';
```

---

### ③ 파티션 메타데이터 등록

```sql
MSCK REPAIR TABLE events;
```

확인 포인트:

- 디렉터리 구조가 파티션으로 정상 인식되는지
- 누락된 파티션 없이 모두 등록되는지

---

## 3. 파티션 기반 조회 테스트 시나리오

```sql
SELECT * FROM events WHERE dt = '2025-01-01';
```

확인 포인트:

- 특정 파티션만 조회되는지
- 불필요한 전체 스캔이 발생하지 않는지

이 테스트를 통해 **파티션 구조가 실제 쿼리 성능과 연결되는지**를 간접적으로 확인한다.

---

## 4. Data Catalog 크롤러 개념

크롤러는 스토리지에 존재하는 데이터를 스캔하여, **테이블 및 파티션 메타데이터를 자동으로 생성·갱신하는 기능**이다.

수동 파티션 관리가 어려운 환경에서 필수적인 구성 요소다.

---

## 5. 크롤러 생성 및 실행 테스트

### ① 크롤러 생성 개념

- 데이터 소스 경로 지정
- 대상 Data Catalog 데이터베이스 설정
- 스케줄 또는 수동 실행 설정

---

### ② 크롤러 실행 후 확인 포인트

- 신규 테이블 자동 생성 여부
- 기존 테이블 스키마 변경 반영 여부
- 신규 파티션 자동 인식 여부

---

## 6. 크롤러 기반 테스트 시나리오 예시

### 시나리오

1. 신규 날짜 파티션 디렉터리 생성
    
    ```bash
    hdfs dfs -mkdir -p /data/events/dt=2025-01-03
    hdfs dfs -put event_0103.csv /data/events/dt=2025-01-03/
    ```
    
2. 크롤러 실행
3. Data Catalog에서 파티션 증가 여부 확인
4. 쿼리로 신규 파티션 조회
    
    ```sql
    SELECT count(*) FROM events WHERE dt = '2025-01-03';
    ```
    

확인 포인트:

- 수동 명령 없이 메타데이터 갱신 여부
- 기존 데이터 영향 없이 신규 데이터만 반영되는지

---

## 7. 테스트 관점 정리

Data Catalog 테스트에서 중요한 것은 다음과 같다.

- 데이터 구조 변경에 대한 추적 가능성
- 파티션 관리 자동화 안정성
- 분석 도구에서의 데이터 탐색 신뢰성

이 검증을 통해 Data Catalog가 단순 조회 도구가 아니라, **데이터 운영의 기준점**으로 활용 가능한지 판단할 수 있다.
