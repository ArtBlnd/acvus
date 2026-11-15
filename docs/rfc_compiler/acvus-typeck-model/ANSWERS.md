# 정의에 대한 답

타입 시스템의 정의에 대해 물어서 받은 답이다. 정의를 모르겠으면 여기를 먼저 보고, 정확한 답이 없으면 묻는다. 바뀔 수 있는 선택은 여기 적지 않는다.

이 모델은 a-mir-formality에서 영감을 받았다. 저쪽은 이미 돌고 있는 타입 시스템을 formalize하고, 이쪽은 처음부터 formalize하고 구현한다.

## 함수

- 어떤 함수든 실행하는 기준은 같다. extern fn은 그 특이 케이스 중 하나다. 내부는 함수마다 다르다.
- **extern fn**은 body가 opaque인 함수다. lambda가 값이 되어 instantiate되기 전의 상태와 같다. capture 같은 제약이 미리 만들어져 있는 상태라, `T`가 parameter에 있으면 bound가 이미 있다.
- **instantiate**는 호출에서 한다. extern fn도 한다.
- **Template**은 함수에 상태가 둘 있어서 있는 개념이다. class의 종류(`Classes<T>`)는 의도한 구분이다.
- **receiver가 있는 호출** `1.plus(2)`는 `plus(1, 2)`와 동치다. receiver의 adjustment는 owned, mut, ref 순서로 바뀌고, 다른 것과 직교다.

## lambda

- **타입**은 Rust와 달리 brand가 없고, signature로만 보인다.
- 밖에서 **opaque**다. 밖에서 아무 정보도 얻지 못한다.
- capture가 정해지기 전까지는 **instantiate 가능한 상태**가 될 수 없다. body가 건드리는 capture 전부를 기다리고, signature에 나오는지와 무관하다.
- **만나는 lambda**는 교집합으로 합쳐진다. 교집합에 맞지 않는 인자는 호출에서 거절된다.

## fallback

- fixpoint에 닿으면 무조건 fallback한다.

## 예

```rust
let f = || 1;
take(f());          // take(u16)
```

거절.

1. `|| 1`은 밖에서 opaque다.
2. 밖에서 아무 정보도 얻을 수 없으니 `1`은 `i32`로 fallback한다.
3. `f()`에서 instantiate한다. `f`는 `() -> i32`다.
4. `take`에서 mismatch.

```rust
let a = 1;
let f = || a;
take(f());          // take(u16)
```

거절이고 `a`는 `i32`다. `f`는 `a`가 정해지기 전까지 instantiate 가능한 상태가 될 수 없고, `a`가 정해지면 `f`의 타입이 정해진다. `a`는 `i32`로 fallback하니 `f`는 `() -> i32`다.

```rust
let mut last = None;
let remember = |x| last = Some(x);
remember(1u8);
```

`last`는 `Option<!>`이다. `x`는 instantiate될 때까지 정해지지 않으니 둘 중 하나다.

1. `last`는 `Option<?0>`에서 `?0`을 기다리고, `remember`는 `last`를 기다린다. 순환이라 `?0`은 결국 `!`로 fallback한다. 그러면 `x`가 될 수 있는 타입은 `!`뿐이다.
2. `last`가 `Option<?0>`인 채로 `remember`를 정하면 순서가 생기고 unsound다. 나중에 `?0`이 `!`가 되어도 free variable인 `x`는 아무 타입으로나 정해진다.

같은 정의에서:

- `let mut a = None; let f = || a = Some(0u8);`은 부르지 않아도 `a`가 `Option<u8>`이다.
- 위 `remember` 앞에 `last = Some(0u8);`이 있으면 `last`는 `Option<u8>`이다. `remember(1u8)`은 성공하고, `remember(1u16)`은 그 인자에서 거절된다.

```rust
let h = if c { |x| 0u8 } else { |x| "a" };
```

부르지 않아도 mismatch로 거절이고, 오류는 `if` 전체다.

```rust
let h = if c { |x| x.to_u32().to_string() } else { |x| x.to_string() };
h(10u32);   // 성공
h(0f32);    // 실패. to_u32()가 없다. 오류는 0f32에서.
```

## 정의가 아직 없는 것

- 무엇이 free인가. 예: 미결정이고, lower bound가 없고, domain이 integer인 class.
- `remember`의 예가 따르는 원리를 어떤 규칙으로 적을지. "bound가 한 개 이상 있어야 한다?" 정도까지만 나왔다.
