# Acvus Template Engine Specification

## Overview

Acvus는 **패턴 매칭과 이터레이션**이라는 단 두 가지 개념 위에 세워진 템플릿 엔진이다.
전통적인 제어문(if, for, while)이 존재하지 않는다.
`=`로 단일 값 패턴 매칭(조건 분기)을, `in`으로 이터레이션(반복)을 표현한다.

## Design Principles

1. **Statement가 없다** — 모든 `{{ }}`는 expression이며, Rust 블럭처럼 자체 output을 가진다.
2. **패턴 매칭이 곧 제어 흐름** — if/for/while 없이 패턴 매칭(`=`)으로 조건 분기를, 이터레이션(`in`)으로 반복을 표현한다.
3. **매칭과 반복의 분리** — `=`는 단일 값 패턴 매칭, `in`은 이터레이션. 서로 다른 메커니즘으로 명확히 구분된다.
4. **함수형 합성** — 파이프 연산자 `|`와 고차 함수로 데이터를 변환한다.
5. **최소 문법** — syntax sugar로 일상적 패턴을 간결하게 유지한다.
6. **모든 변수는 immutable** — `=`는 항상 새 바인딩이다. mutable 변수는 존재하지 않는다.

---

## Syntax

### Literal Text

`{{ }}` 바깥의 텍스트는 그대로 출력된다.

```
안녕하세요, 반갑습니다.
```

→ `안녕하세요, 반갑습니다.`

### Expression

모든 `{{ }}`는 expression이다. 평가 결과가 곧 output이 된다.

```
{{ "hello" }}          → hello
{{ 1 + 2 | to_string }}→ 3
{{ name }}             → (name의 값)
```

출력할 수 있는 타입은 `String`만 가능하다. 다른 타입은 `| to_string`으로 명시적 변환 필요.

### Pipe Operator

`|`로 함수를 체이닝한다. 왼쪽 값이 오른쪽 함수의 첫 번째 인자로 전달된다.

```
{{ list | filter(x -> x != 0) | map(x -> x * 2) }}
```

### Lambda

`->` 화살표 문법으로 익명 함수를 정의한다.

```
x -> x != 0
(x, y) -> x + y
```

### Comment

`{{-- --}}`로 주석을 작성한다. 출력에 포함되지 않는다.

```
{{-- 이것은 주석입니다 --}}
```

---

## Pattern Matching

Acvus의 핵심. 두 가지 키워드로 제어 흐름을 표현한다:

- `=` — 단일 값 패턴 매칭 (조건 분기)
- `in` — 이터레이션 (반복)

하나의 규칙: **패턴 내 상수는 필터하고, 변수는 캡처한다.**

`=`는 항상 새 바인딩을 생성한다. 모든 캡처된 변수는 immutable이다.

### 패턴 매칭 (`=`)

`{{ pattern = expr }}`는 `expr`을 **단일 값으로** 평가하여 `pattern`에 대조한다. 이터레이션은 발생하지 않는다.

```
{{ pattern = expression }}
```

1. `expr`을 평가하여 단일 값을 얻는다.
2. `pattern`을 값에 대조한다.
3. 매칭 성공 시, 패턴 내 변수들이 캡처되고 바디가 렌더링된다.
4. 매칭 실패 시, 다음 arm으로 넘어가거나 `{{_}}`로 넘어간다.
5. `{{/}}`로 블럭을 닫는다.

### 이터레이션 (`in`)

`{{ pattern in expr }}`는 `expr`을 이터레이터로 변환하여 각 원소에 대해 바디를 반복 실행한다.

```
{{ pattern in iterable }}
```

1. `expr`을 평가하여 이터레이터를 얻는다.
2. 이터레이터의 각 원소에 대해 `pattern`을 매칭하고 바디를 렌더링한다.
3. 이터레이터가 비어있으면, `{{_}}`로 넘어간다.
4. `{{/}}`로 블럭을 닫는다.

`in`의 패턴은 irrefutable해야 한다 (변수 바인딩, 오브젝트/튜플 destructuring 등). 리터럴 같은 refutable 패턴은 컴파일 에러.

**조건 분기(`=`)와 반복(`in`)은 별개의 메커니즘이다.**

### 패턴의 종류

#### 변수 패턴

값을 캡처한다. 항상 매칭 성공. 바디가 없는 body-less 바인딩이다.

```
{{ x = $name }}{{ x }}
```

변수 바인딩은 `{{/}}`가 불필요하다. `=`만으로 현재 스코프에 변수를 도입한다.

#### 상수 패턴

값을 필터한다. 캡처하지 않는다.

```
{{ true = is_valid }}
  유효합니다.
{{_}}
  유효하지 않습니다.
{{/}}
```

`is_valid`가 `true`이면 매칭 성공, 바디 1회 렌더링.

#### 복합 패턴

상수로 필터하고, 변수로 캡처한다.

```
{{ [true, a] = pair }}
  {{a}}
{{/}}
```

첫 번째 원소가 `true`인지 필터하고, 두 번째 원소를 `a`에 캡처한다.

#### List Destructuring 패턴

리스트를 분해하여 각 원소를 캡처한다.

```
{{ [a, b, c, ..] = list }}
  첫째: {{a}}, 둘째: {{b}}, 셋째: {{c}}
{{/}}
```

`..`은 나머지 원소를 무시한다. 리스트 길이가 패턴보다 짧으면 매칭 실패.
`..` 위치는 자유: `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.

#### Object Destructuring 패턴

오브젝트에서 필드를 추출한다. Open matching — 패턴에 없는 필드는 무시된다.

```
{{ { name, age, } = $user }}
  {{ name }}
{{/}}
```

오브젝트 패턴은 trailing comma가 필수이다.

#### Tuple 패턴

튜플을 분해하여 각 원소를 캡처한다. `_`로 특정 원소를 무시할 수 있다.

```
{{ (name, age) = $pair }}
  {{ name }}
{{/}}

{{ (name, _) = $pair }}     ← 두 번째 원소 무시
  {{ name }}
{{/}}
```

Tuple은 고정 길이이며, List와 달리 `..` rest 패턴을 지원하지 않는다.
패턴의 원소 수와 튜플의 원소 수가 일치해야 한다 (불일치 시 컴파일 에러).

#### Range 패턴

범위 내에 있는지 테스트한다.

```
{{ 0..10 = $age }}
  어린이
{{_}}
  성인
{{/}}
```

Range 바운드는 Int만 허용한다.

### Multi-arm

하나의 expression에 대해 여러 패턴을 순차적으로 매칭할 수 있다.
첫 arm에서 `{{ pattern = expr }}`로 매칭 대상을 지정하고,
이후 arm에서는 `{{ pattern = }}`으로 작성하면 동일한 expression에 대한 추가 arm이 된다.
`=` 뒤에 expression이 없으면 continuation arm, 있으면 새로운 binding이다.

```
{{ "admin" = role }}
  관리자 페이지
{{ "user" = }}
  사용자 대시보드
{{_}}
  게스트 뷰
{{/}}
```

#### Tuple을 활용한 Multi-arm

여러 값을 동시에 매칭할 때 Tuple 패턴을 사용한다:

```
{{ (0, 1) = ($a, $b) }}
  a=0, b=1
{{ (0, _) = }}
  a=0
{{ (1, 1) = }}
  a=1, b=1
{{ (_, _) = }}
  기타
{{/}}
```

중첩 match 대신 flat한 tuple 매칭으로 가독성을 높일 수 있다.

### Iteration (`in`)

`in` 키워드로 iterable을 반복한다. `=`와 달리 이터레이터를 생성하여 각 원소에 대해 바디를 실행한다.

```
{{ x in $items }}
  {{ x | to_string }}
{{/}}

{{ { name, } in $users }}
  {{ name }}
{{/}}

{{ (a, b) in $pairs }}
  {{ a }}: {{ b | to_string }}
{{/}}
```

`in`의 패턴은 irrefutable해야 한다:
- **허용**: 변수 바인딩 (`x`), 오브젝트 destructuring (`{ name, }`), 튜플 destructuring (`(a, b)`)
- **불허**: 리터럴 (`"admin"`), 레인지 (`0..10`) — 컴파일 에러

Multi-arm은 `in`에서 사용할 수 없다. 조건부 필터가 필요하면 `filter`를 파이프로 사용한다:

```
{{ x in $items | filter(x -> x != 0) }}
  {{ x | to_string }}
{{/}}
```

### Catch-all `{{_}}`

`{{_}}`는 패턴 매칭이 실패했거나 이터레이터가 비었을 때 실행된다.
모든 match/iter 블럭에 `{{_}}`가 필수이다 (단, 변수 바인딩과 스토리지 쓰기는 예외).

```
{{ true = $is_valid }}
  유효합니다.
{{_}}
  유효하지 않습니다.
{{/}}

{{ item in $list }}
  {{ item }}
{{_}}
  리스트가 비어있습니다.
{{/}}
```

### Close Block `{{/}}`

`{{/}}`로 match/iter 블럭을 닫는다. indent modifier를 붙일 수 있다.

| 문법 | 의미 |
|------|------|
| `{{/}}` | 기본 닫기 |
| `{{/+N}}` | indent를 N만큼 증가 |
| `{{/-N}}` | indent를 N만큼 감소 |

```
{{ item in $list }}
  {{ item }}
{{/+2}}
```

---

## Storage Reference (`$`)

`$`는 외부 스토리지에 대한 레퍼런스를 나타낸다. 모든 변수는 immutable이지만,
`$`는 스토리지 slot에 값을 쓸 수 있다.

### 규칙

1. **`$`는 `$`에서만 올 수 있다** — 무에서 `$` 변수를 생성할 수 없다. 반드시 외부에서 제공된 `$`로부터 파생되어야 한다.
2. **`{{ $name = expr }}`는 스토리지 쓰기** — `$name`이 가리키는 스토리지 slot에 `expr`의 값을 쓴다. `{{/}}` 불필요.
3. **Destructuring으로 `$` 전파** — `$` 변수를 destructure할 때, `$`로 꺼내면 레퍼런스, `$` 없이 꺼내면 값 복사.

```
{{ $global }}                ← 스토리지 값 출력
{{ $global.field }}          ← 필드 값 출력
{{ $global = expr }}         ← 스토리지 전체에 쓰기
{{ { $value, name, } = $global }}
  {{ $value = 42 }}          ← $global.value에 쓰기
  {{ name }}                 ← immutable 복사본
{{/}}
```

---

## Type System

### 원칙

- **모든 타입은 컴파일 타임에 결정된다.** 동적 타입은 존재하지 않는다.
- 변수의 타입은 항상 RHS expression의 타입으로 결정된다.
- Storage(`$`) 변수의 타입은 외부에서 제공된다.
- 유저가 타입을 명시할 필요가 없다. 전부 추론된다.

### 타입 종류

| 타입 | 예시 |
|------|------|
| `Int` | `42` |
| `Float` | `3.14` |
| `String` | `"hello"` |
| `Bool` | `true`, `false` |
| `List<T>` | `[1, 2, 3]` |
| `Object<{...}>` | `{ name, value, }` |
| `Tuple<T1, T2, ...>` | `(1, "hello")` |
| `Range` | `0..10`, `0..=10` |
| `Fn(A, B) -> C` | `x -> x + 1` |

### 타입 규칙

#### 산술 연산
- **`Int op Int` → `Int`**, **`Float op Float` → `Float`**. 혼합 연산(`Int + Float`)은 **타입 에러**.
- **`Int / Int` → `Int`** (truncate). 정수 나눗셈은 항상 버림.
- 명시적 변환이 필요: `n | to_float`, `x | to_int`.

#### Null과 List 통합
- 별도의 `Option<T>` 타입은 존재하지 않는다.
- Null은 `List<T>`의 빈 리스트 `[]`로 표현된다.
- 단일 값은 `[value]` (1-원소 리스트)로 취급된다.

#### List
- **동질적(homogeneous)**: `[1, "a"]`는 컴파일 에러.
- 빈 리스트 `[]`의 타입은 사용 컨텍스트에서 추론된다. 단독 사용은 타입 에러.

#### Tuple
- **이종(heterogeneous)**: `(1, "hello")`는 `Tuple<Int, String>`.
- **고정 길이**: 타입에 원소 수가 포함된다.
- **비반복(non-iterable)**: 패턴 매칭으로 destructure만 가능, 이터레이션 불가.
- `_`로 특정 원소를 무시할 수 있다 (패턴 컨텍스트에서만).

#### Object 매칭
- Object는 **open (서브셋 매칭)**이다.
- `{ name, }` 패턴은 `{ name, age, email }` 오브젝트와 매칭된다.

#### Range 바운드
- **`Int`만 허용**. `Float` range는 타입 에러.

#### Emit (출력)
- `{{ expr }}`로 출력할 수 있는 타입은 **`String`만** 가능.
- 다른 타입은 명시적 변환 필요: `{{ count | to_string }}`.

#### Exhaustiveness
- **`{{_}}` catch-all은 필수**. 모든 match/iter 블럭에 `{{_}}`가 있어야 한다.
- 정적 exhaustiveness 분석이 불필요해진다. 컴파일러는 `{{_}}` 존재 여부만 검사.

#### 동등성 비교 (`==`)
- 모든 타입에서 구조적 비교(deep equality) 지원.
- Mutation 없음 + 순환 참조 불가능이므로 구조적 비교는 항상 sound.

### 빌트인 제네릭

유저 정의 제네릭은 없지만, 빌트인 함수는 내부적으로 제네릭 시그니처를 가진다.
타입 체커가 호출 시점에 unification으로 구체 타입을 결정한다.

```
filter: (List<T>, Fn(T) -> Bool) -> List<T>
map:    (List<T>, Fn(T) -> U) -> List<U>
```

유저는 타입 변수를 볼 일이 없다. 파이프가 암묵적으로 제네릭을 제공한다.

---

## Compilation Pipeline

```
Template → AST → Type Check → MIR → Backend (미정)
```

- **AST**: 구문 레벨 표현 (파서 출력)
- **Type Check**: 타입 추론 및 검증
- **MIR**: 타입이 붙은 linear IR. 패턴 매칭 lowering, 파이프 desugar, 클로저 캡처 분석 포함.
- **Optimization**: MIR → MIR 변환 패스.
- **Backend**: MIR에서 최종 타겟으로 코드 생성. 후보: WASM, 네이티브, JS 등.

### 호출 구분

- **`call`**: 빌트인 함수. 동기 실행.
- **`async_call`**: 외부 함수. 비동기 실행. `await` 필요.

---

## Async & Parallelism

- **외부 함수 호출은 모두 async로 취급**한다.
- 빌트인 함수(산술, `filter`, `map` 등)는 내부에서 동기 실행된다.
- Mutation이 없으므로 순수 함수는 컴파일러가 자동으로 병렬화할 수 있다.

### `map` vs `pmap`

| 함수 종류 | `map` | `pmap` |
|-----------|-------|--------|
| **pure** | 옵티마이저가 자동 병렬화 | 동일 결과 |
| **effectful** | 순차 실행 (기본값) | 유저가 명시적으로 병렬화 |

```
{{-- 순차 실행 --}}
{{ user_ids | map(id -> fetch_user(id)) }}

{{-- 유저가 명시적으로 병렬화 --}}
{{ user_ids | pmap(id -> fetch_user(id)) }}
```

---

## Design Decisions

| 결정 | 값 | 이유 |
|------|-----|------|
| 매칭 vs 반복 | `=` / `in` 분리 | `=`는 단일 값 대조, `in`은 이터레이션. 명확한 의도 표현 |
| 서브타이핑 | 없음 | 복잡도 회피 |
| 유저 정의 제네릭 | 없음 | 파이프 + 빌트인이 암묵적으로 제공 |
| Mutation | 없음 | 모든 바인딩은 immutable |
| 리스트 | Homogeneous | Heterogeneous 허용 시 런타임 태그드 유니온 필요 |
| 튜플 | Heterogeneous, 고정 길이 | 여러 값을 하나로 묶어 매칭, `_` 와일드카드 지원 |
| 추론 방식 | 단방향 + 빌트인 unify | 빌트인 함수만 내부적으로 타입 변수 사용 |
| 인덱스 접근 | 별도 문법 없음 | `enumerate` 같은 함수로 처리 |
| include/import | 없음 | 단일 템플릿이 완결적 |
| enum | 없음 | 호스트가 데이터를 평탄화하여 전달 |
| 함수 정의 | 호스트 사이드만 | 템플릿에서는 파이프 + 람다 컴포지션만 |

---

## Syntax Summary

| Syntax | Description |
|---|---|
| `text` | 리터럴 텍스트, 그대로 출력 |
| `{{ expr }}` | expression 평가 후 출력 |
| `{{ pattern = expr }}...{{/}}` | 패턴 매칭 블럭 (단일 값 대조) |
| `{{ pattern in expr }}...{{/}}` | 이터레이션 블럭 (반복) |
| `{{ x = expr }}` | 변수 바인딩 (body-less, `{{/}}` 불필요) |
| `{{ pattern = }}` | multi-arm continuation: 이전 매칭 대상에 대한 추가 arm |
| `{{_}}` | catch-all arm (매칭 실패 또는 이터레이터 비었을 때) |
| `{{/}}` `{{/+N}}` `{{/-N}}` | 블럭 닫기 (indent modifier 선택) |
| `{{ $name = expr }}` | 스토리지 쓰기 (body-less) |
| `{{ { $a, b, } = $x }}` | destructuring: `$a`는 레퍼런스, `b`는 값 복사 |
| `expr \| func(args)` | 파이프 연산자 |
| `x -> expr` | 람다 expression |
| `[a, b, ..]` | list destructuring 패턴 |
| `{ name, age, }` | object destructuring 패턴 (trailing comma 필수) |
| `(a, b)` | tuple expression / 패턴 |
| `(a, _)` | tuple 패턴 (wildcard) |
| `0..10` `0..=10` `0=..10` | range expression / 패턴 |
| `{{-- comment --}}` | 주석 (출력 없음) |
