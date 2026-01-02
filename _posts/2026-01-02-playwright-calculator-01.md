---
layout: posts
title: "[Playwright] 요금 계산기 테스트 - 기본 시나리오"
categories: ['sqa']
tags: [playwright]
related: false
---
<br/>

## 1. 테스트 대상 개요

### 대상 페이지

- KakaoCloud 요금 계산기
    
    https://www.kakaocloud.com/pricing/calculator
    

### 테스트 목적

사용자가 실제로 요금 계산기를 사용하는 흐름을 그대로 자동화한다.

### 테스트 시나리오

1. 요금 계산기 페이지 진입
2. **Virtual Machine 서비스 추가**
3. VM 설정값 선택
    - 운영체계(OS)
    - 인스턴스 타입
    - 볼륨 사용량(SSD)
    - 인스턴스 개수
4. 서비스 목록에 담기
5. 상세 내역 패널 열기
6. 테이블에 입력한 값이 정확히 노출되는지 검증

---

## 2. 전체 테스트 구조

Playwright **Page Object Model (POM)** 기반으로 구성했다.

```
Calculator/
├─ pages/
│  ├─ pricing.page.ts        // 요금 계산기 메인 페이지
│  ├─ vm.config.panel.ts     // VM 설정 카드
│  └─ estimate.drawer.ts    // 상세 내역 패널
│
├─ tests/
│  └─ vm.pricing.spec.ts    // 실제 테스트 시나리오
│
├─ playwright.config.ts
└─ package.json
```

### 구조 설계 원칙

- **페이지 단위로 책임 분리**
- 테스트 파일에는 **행위 흐름만**
- locator / UI 변경 영향은 Page Object에 격리

---

## 3. Page Object 설계 및 역할

---

### 3.1 pricing.page.ts

**요금 계산기 메인 페이지**

### 책임

- 요금 계산기 페이지 진입
- Virtual Machine 카드 열기

### 핵심 포인트

- “페이지 진입 성공”의 기준은 **헤딩 텍스트**
- 서비스 버튼은 role 기반 locator 사용

```tsx
import { Page, expect } from '@playwright/test';

export class PricingPage {
  constructor(private readonly page: Page) {}

  async open() {
    // config 파일에 baseURL: 'https://www.kakaocloud.com' 설정해야됨
    await this.page.goto('/pricing/calculator');
    
    // 요금계산기 페이지 로딩 확인
    await expect(
      this.page.getByLabel('text', { exact: true }).nth(5)
    ).toBeVisible();
  }

  async clickVirtualMachine() {
    await this.page
      .locator('button').filter({ hasText: 'Virtual Machine' })
      .click();
  }
}
```

---

### 3.2 vm.config.panel.ts

**Virtual Machine 설정 카드**

### 책임

- VM 카드 내부 옵션 선택
- 서비스 목록에 담기 버튼 클릭

### UI 특징 (중요)

- Radix 기반 드롭다운
- label이 아니라 **텍스트 + 버튼 구조**
- role 기반 접근이 가장 안정적

### 구현 전략

- `paragraph(운영체계)` → 같은 카드 내 `button "선택"` 클릭
- option은 role=option 기반 선택

```tsx
import { Page, expect } from '@playwright/test';

export class VMConfigPanel {
  constructor(private readonly page: Page) {}

  async selectOS(osName: string) {
    // 드롭다운 열기
    await this.page
    .getByText('운영체계 (OS)')
    .locator('xpath=following::button[1]')
    .click();
  
    // 옵션 나타날때까지 대기 
    const option = this.page.getByText(osName, { exact: true });
  
    // 옵션 선택
    await option.waitFor({ state: 'visible' });
    await option.click();
  }

  async selectInstanceType(typeName: string) {
    await this.page
    .getByText('인스턴스 타입')
    .locator('xpath=following::button[1]')
    .click();

    const option = this.page.getByText(typeName, { exact: true });
  
    await option.waitFor({ state: 'visible' });
    await option.click();
  }

  async setVolumeSSD(size: number) {
    const volumeInput = this.page
    .getByRole('textbox', { name: 'GiB 이상 입력' })
    .locator('xpath=ancestor::*[self::div or self::li][1]');
    
    await volumeInput
    .getByRole('textbox')
    .fill(size.toString());
  }

  async setInstanceCount(count: number) {
    const countInput = this.page
    .getByRole('textbox', { name: '0' })
    .locator('xpath=ancestor::*[self::div or self::li][1]');
    
    await countInput
    .getByRole('textbox')
    .fill(count.toString());
  }

  async addServiceToList() {
    const addButton = this.page
    .locator('button')
    .filter({ hasText: '서비스 목록에 담기' });
    
    await expect(addButton).toBeEnabled();
    await addButton.click();
  }
}
```

---

### 3.3 estimate.drawer.ts

**상세 내역 패널**

### 가장 많이 틀렸던 부분

❌ 잘못된 접근

- `Virtual Machine`을 테이블의 부모로 가정

<img width="962" height="573" alt="image" src="https://github.com/user-attachments/assets/b2ad6d99-0a05-4b30-8e06-63acd965d6f9" />


✅ 올바른 접근

- **패널 전환 기준**
    - `예상 요금 내역` 텍스트
- **테이블 자체 기준**
    - role=table
    - 컬럼 존재 여부

### 구현

```tsx
import { Page, expect } from '@playwright/test';

export class EstimateDrawer {
  constructor(private readonly page: Page) {}

  async openDetail() {
    await this.page
    .locator('button').filter({ hasText: '상세 내역 보기' })
    .click();

    // 상세 내역 패널 전환 완료 신호
    await expect(this.page.getByText('예상 요금 내역', { exact: true }))
    .toBeVisible();

  }

  async expectVirtualMachineRow(params: {
    os: string;
    instanceType: string;
    volumeSSD: string;
    instanceCount: string;
  }) {
    const table = this.page.getByRole('table');

    // VM 테이블임을 보장
    await expect(
      table.getByText('운영체계 (OS)')).toBeVisible();

    const row = table
      .getByRole('row')
      .filter({ hasText: params.os });

    await expect(row).toBeVisible();
    await expect(row).toContainText(params.instanceType);
    await expect(row).toContainText(params.volumeSSD);
    await expect(row).toContainText(params.instanceCount);
    await expect(row).toContainText('원');
  }
}
```

<img width="1199" height="752" alt="image" src="https://github.com/user-attachments/assets/969bda02-e94f-4b00-a509-9c5ddfb1cfe4" />


---

## 4. 실제 테스트 파일 (vm.pricing.spec.ts)

### 테스트 흐름만 남기고 최대한 단순화

```tsx
import { test } from '@playwright/test';
import { PricingPage } from '../../pages/pricing.page';
import { VMConfigPanel } from '../../pages/vm.config.panel';
import { EstimateDrawer } from '../../pages/estimate.drawer';

test('Virtual Machine 요금 계산 시나리오', async ({ page }) => {
  const pricingPage = new PricingPage(page);
  const vmConfigPanel = new VMConfigPanel(page);
  const estimateDrawer = new EstimateDrawer(page);

  // 1. 요금 계산기 페이지 진입
  await pricingPage.open();

  // 2. Virtual Machine 서비스 선택
  await pricingPage.clickVirtualMachine();

  // 3. VM 설정
  await vmConfigPanel.selectOS('Ubuntu 20.04');
  await vmConfigPanel.selectInstanceType('m3az.large (2vCPU / 8GiB Memory)');
  await vmConfigPanel.setVolumeSSD(100);
  await vmConfigPanel.setInstanceCount(2);

  // 4. 서비스 목록에 담기
  await vmConfigPanel.addServiceToList();

  // 5. 상세 내역 확인
  await estimateDrawer.openDetail();
  await estimateDrawer.expectVirtualMachineRow({
    os: 'Ubuntu 20.04',
    instanceType: 'm3az.large',
    volumeSSD: '100',
    instanceCount: '2',
  });
});
```

<img width="990" height="231" alt="image" src="https://github.com/user-attachments/assets/7509fb44-13ad-4c63-a55f-be1f0d019161" />


---

## 5. 테스트 회고

### 1️⃣ UI 구조를 추측하지 말고 **ARIA 기준으로 해석**

- 텍스트가 부모일 거라는 가정 ❌
- role, column, row 구조가 진짜 기준

### 2️⃣ “화면 전환”은 반드시 명시적으로 검증

- 버튼 클릭 ≠ 화면 전환
- **전환을 증명하는 텍스트 하나를 잡아라**

### 3️⃣ 테이블 검증

- table → row → text
- 절대 name 하나로 끝내지 않는다
