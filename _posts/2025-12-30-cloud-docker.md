---
layout: posts
title: "[Cloud Service] Docker 활용"
categories: ['cloud']
tags: [docker]
---

클라우드 서비스 테스트에서 Docker는 단순한 개발 도구가 아니라, **애플리케이션 배포 단위가 실제로 어떻게 이동·실행되는지를 검증하는 핵심 수단**이다.

SQA 관점에서 Docker를 사용하는 목적은 다음과 같다.

- 동일한 이미지를 로컬과 클라우드 환경에서 일관되게 사용할 수 있는지
- 이미지 push / pull 과정이 정상적으로 동작하는지
- VM 환경에서 컨테이너가 의도한 설정으로 실행되는지

아래는 **일반적인 Docker 기반 테스트 흐름**을 정리한다.

---

## 1. Docker Desktop 설치 (로컬 환경)

로컬 테스트 환경에서는 Docker Desktop을 사용해 Docker 엔진과 CLI를 함께 구성한다.

설치 후 가장 먼저 확인하는 것은 Docker가 정상적으로 실행되는지 여부다.

```bash
docker version
docker info
```

- Client / Server 정보가 모두 출력되면 정상
- 이 단계에서부터 로컬 환경 자체가 테스트 대상이 된다

---

## 2. Docker Hub 개념 및 역할

Docker Hub는 Docker 이미지를 저장하고 공유하는 공개 레지스트리다.

클라우드 테스트 관점에서 Docker Hub는 다음 역할을 한다.

- 이미지 push 대상 레지스트리
- 클라우드 VM에서 이미지 pull 테스트용 저장소

실제 서비스에서는 프라이빗 레지스트리를 사용하더라도, **동작 방식은 동일**하기 때문에 개념 검증에 적합하다.

---

## 3. 로컬 환경에서 Docker 컨테이너 실행

이미지가 정상적으로 실행되는지 확인하는 것은 가장 기본적인 검증 단계다.

```bash
docker run hello-world
```

또는 임의의 테스트 이미지 실행:

```bash
docker run -d -p 8080:80 nginx
```

확인 포인트:

- 컨테이너가 정상적으로 기동되는지
- 포트 바인딩이 의도대로 동작하는지

---

## 4. Docker Hub 로그인

이미지를 push하기 위해서는 Docker Hub 계정으로 로그인해야 한다.

```bash
docker login
```

- 정상 로그인 여부
- 잘못된 계정 정보 입력 시 에러 처리

이 과정 또한 인증 관점에서 하나의 테스트 포인트가 된다.

---

## 5. Docker 이미지 생성 및 태깅

테스트용 Dockerfile을 기반으로 이미지를 생성한다.

```bash
docker build -t username/test-image:latest .
```

이미지 목록 확인:

```bash
docker images
```

태깅 규칙이 올바르게 적용되었는지 확인한다.

---

## 6. Docker 이미지 Push (로컬 → 레지스트리)

로컬에서 생성한 이미지를 Docker Hub로 push한다.

```bash
docker push username/test-image:latest
```

확인 포인트:

- 이미지 업로드 정상 여부
- 권한 문제 발생 여부
- 네트워크 오류 시 재시도 처리

이 단계는 **이미지 전송 안정성**을 검증하는 과정이다.

---

## 7. 클라우드 VM에서 Docker 이미지 Pull

이제 클라우드 VM에 SSH 접속한 상태에서 이미지를 pull한다.

```bash
docker pull username/test-image:latest
```

확인 포인트:

- 레지스트리 접근 가능 여부
- 네트워크 / 방화벽 설정 영향
- 이미지 무결성 문제 여부

---

## 8. 클라우드 VM에서 컨테이너 실행 확인

Pull한 이미지를 실제로 실행해본다.

```bash
docker run -d test-image
```

또는 포트 바인딩 테스트:

```bash
docker run -d -p 8080:80 username/test-image:latest
```

확인 포인트:

- 컨테이너 정상 기동 여부
- VM 리소스 사용 변화
- 서비스 접근 가능 여부

---

## 9. 테스트 관점 정리

- 로컬 → 레지스트리 → 클라우드 VM 흐름 검증
- 동일 이미지의 환경 간 일관성 확인
- 이미지 배포 실패 지점 식별 가능
