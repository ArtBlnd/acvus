# typeck 기점 (working note)

진행 중인 방향과 열린 질문을 적는다. 결정된 것은 `docs/transparency-constraint-type-system.md`(KR: `-kr.md`)에 있고, 이 노트는 그것을 대신하지 않는다. 줄 번호는 재구성 전 EN spec, 곧 `transparency-constraint-type-system.md.old`(393줄) 기준이다.

## 1. 문제

미지수를 정하는 결정이, 그 미지수에 걸린 run의 제약을 다 보지 못한 채 내려진다. spec의 Transparency 머리 문단은 이것을 금지한다. 지금의 Lifecycle step들은 아직 그 문단을 지키지 않는다(5절 8).

## 2. 정의

- **세 결정은 하나다.** Unify(l.46, `S = lb(α)`, `|lb| = 1`), Join(l.107, `S = R(α)`), Fallback 1(l.157, `S = ∅`, A2에 따라 `lb = ∅`이거나 `R = ∅`)은 모두 `α ≐ ⊔S`다. `!`가 ⊔의 항등원(l.120)이라 `⊔∅ = !`다. 셋은 S의 크기만 다르다.
- **보류된 제약(suspended constraint).** 기다리는 대상(waiting target: Join, coercion, use)에 걸려 있어서 아직 기록되지 않은 run의 제약. semi callee의 use에는 그 C가 보류된다(spec Transparency 머리 문단).
- **σ\*.** run의 제약 전체(기록된 것과 보류된 것)의 점별 ≤-최소해. membership `T ∈ S`가 들어가면 "기록된 lower bound 위에 있는, S 안의 최소 타입"이다. 후보가 둘 이상 남은 use 때문에 없을 수 있다.
- **I\*.** `α ≐ ⊔S`는 `⊔S = σ*(α)`일 때만 내린다. 곧 결정을 바꿀 수 있는 보류된 제약이 α에 걸려 있지 않을 때만 내린다.
- **결정을 바꾸지 않는 보류된 제약.** α가 제약의 아래쪽 covariant 자리에만 있는 경우. 그 제약은 α를 위에서만 막고, `!`는 ≤의 bottom이다(l.120, l.136).
- **H.** 결정을 바꿀 수 있는 보류된 제약이 걸려 있지 않은 Fallback 1 대상의 최대 집합. H 전체를 `!`로 정해도 I*가 지켜진다.
- **형태.**
  - 순서: H를 먼저 정하면 풀린다(E5).
  - 순환: 기다리는 대상 w가 x를 기다리면서(`w ⇢⁺ x`), w에 보류된 제약이 x의 결정을 바꾼다. 어떤 순서로도 풀리지 않는다(F14, E8, R 원소의 invariant part).

## 3. 증인

| 증인 | 제약이 보류된 곳 | 보류된 제약 |
|---|---|---|
| F14 `let mut v = Vec::new(); v.push(1u8);` | `push` use (opaque) | receiver의 `?t = T`, 인자 `u8`이 `T`와 만남 |
| E8 `let mut s = None; while c { s = step(s); }` | `step` use (opaque) | `r = Option⟨u8⟩`가 α_s의 lower bound |
| E5 `last = Some(read())` 루프 뒤 `if let Some(v) = last { show(v) }` | `last`의 Join | `?v ≥ u8` (part별 관계) |
| E4 `let scale = \|x\| x * 3u8; scale(10)` | `scale` use (semi) | C의 `X * u8`가 `X`를 `u8`로 제약 |
| F7a `let remember = \|x\| last = Some(x); remember(1u8)` | `remember` use (semi) | C 안의 capture 쓰기가 caller의 `last` 타입과 meeting |
| `fill(&mut buf)`, `fill(&mut Vec⟨u8⟩)` | coercion (opaque 경계) | invariant 자리의 `?t = u8` |
| `if c { &a } else { &b }`, `a: Vec⟨?t⟩`, `b: Vec⟨u8⟩` (규칙에서 도출) | 결과의 Join | R 원소의 invariant part `?t = u8` |

F7b(`*slot = Some(1u8)`)는 B(Deref·Elem을 거친 쓰기를 meeting으로) 이후 보류된 제약이 아니다. 지금 Fallback으로도 `Option⟨u8⟩`로 풀린다.

## 4. 도출된 성질

- 결정이 모두 σ\*의 값이면, 결정을 더해도 σ\*는 여전히 해이고 최소다. 그래서 결정 순서가 결과를 바꾸지 않고, l.23의 "되돌리지 않음"도 저절로 지켜진다.
- 기다리는 변수와 상관없이 모든 해에서 성립하는 제약은, 기록해도 해를 지우지 않는다.
- 따로 푸는 callable의 run(Instantiate 1)과 instance run(Instantiate 2)은 같은 제약의 경계를 같게 판정한다. 따로 푸는 run의 미지수는 instance run이 시작할 때도 정해지지 않았고, universal인 것은 instance run이 시작할 때 정해져 있다.
- 순서는 soundness를 건드리지 않는다. 결정 뒤의 검사가 남아 있기 때문이다: l.104–107의 re-fire, coercion 표 검사, 후보가 정확히 하나, 남은 것은 거부. 이 결론은 l.106의 re-fire가 무조건 일어난다는 것에 기댄다.

## 5. 열린 것

1. **기점.** 보류된 제약 중 무엇을 결정 전에 기록하는가.
   - 후보가 둘 이상 남은 use: 모든 후보에 공통인 부분.
   - semi의 C: `T`에서 끝까지 풀지 않고도 성립하는 부분. 재귀 group에서 이게 끝나는지가 가장 어렵다.
2. **membership `T ∈ S`.** run의 제약 종류로 둘지. opaque의 S를 선언이 적는 C로 둘지(semi는 추론된 C). Fallback이 S 안에서 고르게 할지.
3. **rank(l.179).** caller의 미지수인 parameter: `T ∈ S`가 가능하면 닿고, rank에는 아무것도 더하지 않는다. 원문 문장은 기점과 함께 정한다.
4. **part 단위 경계.** 바깥은 선언이 주고 안쪽은 caller가 고르는 parameter(`Option⟨T⟩` 등)를 part마다 판정할지. 기점과 함께 정한다.
5. **Mark(l.327)의 "each coercion included".** meeting으로 들어올 때의 변환도 포함하는가.
6. **`Elem`을 B에 넣은 것과 indexing 전제가 맞는가.**
7. **Fallback 2.** 최소해 원리에 맞지 않는다. 가정: numeric 타입은 extern-defined가 아니라서, 정수 타입끼리는 ≤ 순서가 없다. 따로 기준을 세워야 한다.
8. **Transparency 머리 문단과 충돌하는 Lifecycle step.**
   - ~~Fallback guard(l.156)는 나가는 ⇢만 읽는다.~~ Fallback 1에 `σ(α) = !` 조건을 더해 정리됨(l.157). 결과: E5는 받아들여지고, F14·F7a·Example 3의 loop `push`는 틀린 `!` 대신 "안 풀린 채 남음"(l.23)으로 거부된다. Fallback 2는 7번으로 남는다.
   - Join(l.104–107)은 R만 본다. D 원소에 보류된 제약을 보지 않는다(E8). 또 D/R 나눔이 하나로 정해지지 않는다(E9).
   - Unify(l.46)는 보류된 제약을 보지 않는다.
   - Instantiate 2(l.76)는 한 방향이다.

## 6. constraint graph (준량의 해법)

### 정의

- 동치류는 kind와 함께, 그 타입에 걸린 제약을 지닌다: 하한, 등식, 경계에서의 coercion(`T ⇝ U`), use.
- constraint graph는 동치류들과 그 제약 전체다. 노드는 동치류이고, 간선은 제약이다.

### 해법

1. **기록.** run 동안 body의 텍스트가 만드는 제약을 constraint graph에 기록한다. 아무것도 고르지 않고, back-prop은 없다.

   | 텍스트 | 기록되는 것 |
   |---|---|
   | meeting | 하한 `τ ≤ α` |
   | invariant 자리 | 등식 |
   | 경계 | `T ⇝ U` |
   | use | 후보 집합. 후보가 하나면 그 signature |
   | projection | kind |

   unify는 등식을 기록하는 것이다: 두 동치류를 합치고, kind를 재귀적으로 unify한다(l.52). 서로 unify되지 않는 두 등식은 거부한다.
2. **Join (정방향).** run 끝에 동치류마다 값을 위에서 아래로 고른다. meeting과 경계가 아닌 자리의 등식만 쓰고, 경계는 쓰지 않는다. 이 값이 σ다.
   - 하한은 `τ ≤ α`라는 제약이다. 등식이 `α = U`를 주면 하한은 `τ ≤ U`가 된다. 등식이 없으면 값은 하한들의 최소해, 곧 ⊔다.
   - constructor를 고르면, 하한들의 part 사이 관계(l.112)를 part의 동치류에 기록한다: covariant part는 하한, invariant part는 등식. 그래서 바깥 동치류를 먼저 고르고, 거기서 오는 기록을 다 받은 뒤 part를 고른다.
   - 하한끼리만 도는 순환(`x ≥ y`, `y ≥ x`)은 한 동치류로 합친다(cycle elimination).
   - use나 constructor를 거치는 순환은 SCC마다, SCC 밖에서 온 하한의 ⊔ `x₀`에서 시작해 use를 한 번 고른다. 결과가 해이면 받아들이고, 아니면 거부한다.
3. **경계 (역방향).** Join이 아무것도 정하지 않은 part, 곧 제약이 없거나 `!`이거나 literal인 part에만, 거기 걸린 경계의 타입이 등식(invariant)으로 들어간다. 그다음 다시 Join한다. 한 part에 경계들이 서로 다른 타입을 주면 거부한다: 하나를 고르고 나머지를 coerce하는 방법이 여럿이라 해가 하나로 정해지지 않는다.
4. **Coercion.** 정해진 타입에 맞지 않는 값이 들어오는 자리마다, 경계든 meeting이든, 그 자리에서 conversion fn을 호출한다. fn이 없으면 거부한다. 타입은 고르지 않는다.

### 성질

- 단조성: 기록만 하고 지우지 않는다.
- 비순차성: 끝의 결과는 constraint graph만의 함수다.
- 에러 위치: 충돌하면 관련된 제약의 출처를 모두 짚을 수 있다.
- 검증: 모든 동치류의 해가 정해진 뒤, 동치류마다 국소적으로 검증된다.
  - 기록된 제약이 모두 해에서 성립하는가: soundness.
  - 값이 SCC 밖에서 온 하한의 ⊔, 또는 등식이 준 타입과 같은가: 최소해.
- 푸는 단위: 순환 밖의 동치류는 하나씩, 순환 안의 동치류는 SCC마다 함께, 의존 순서대로 푼다.
- 덤프: constraint graph를 덤프하면 동치류마다 그 제약과 출처가 보인다.

### 도출된 것

- I\*가 저절로 지켜진다. 보류된 제약이 없어지니, 기록할 수 있는 제약에 대해서는 기점(5절 1)이 풀린다.
- D/R이 필요 없다(5절 8의 Join).
- 옛 Lifecycle의 Unify step(`lb(α) = {τ} ⟹ α ≐ τ`)은 하한이 하나인 Join이라 해법 2에 들어간다(5절 8의 Unify).
- Join이 하는 기록은 언제나 아래, 곧 아직 고르지 않은 part의 동치류로 간다. 그래서 고른 것을 건드리지 않는다.
- 하한을 검사가 아니라 제약으로 두어야 F14가 풀린다. 검사로만 두면 `T₁`에 제약이 없어 `σ(T₁) = !`이고, `Vec⟨!⟩ ≤ Vec⟨u8⟩`가 거짓이다.
- part를 바깥보다 먼저 고르면 `if c { &a } else { &b }`(`a: Vec⟨?t⟩`, `b: Vec⟨u8⟩`)에서 `?t = !`가 되어 거부된다. 위에서 아래로 고르면 결과를 `&Vec`으로 고를 때 `?t = u8`이 기록된다.
- 증인:

  | 증인 | 기록되는 제약 | 끝의 join | 결과 |
  |---|---|---|---|
  | F14 | `v ≥ Vec⟨T₁⟩`, receiver에서 `Vec⟨T₁⟩ = Vec⟨T₂⟩`(invariant), `T₂ ≥ u8` | `T₁ = T₂ = u8` | 받아들임 |
  | E8 | `s ≥ Option⟨?k⟩`, `s ≥ r`, `r = Option⟨u8⟩`(`step`의 signature), `s ⇝ Option⟨u8⟩` | `s = Option⟨! ⊔ u8⟩ = Option⟨u8⟩` | 받아들임 |
  | E5 | `last`의 하한들, `?v ≥ u8` | `?k = !`, `?v = u8` | 받아들임 |
  | `let mut x = 1u8; while c { x = x * 2; }` | `x ≥ u8`, `x ≥ r`, `r = res(x * ?l)` | SCC `{x, r}`: `x₀ = u8`, 한 걸음에 `r = u8 ≤ x₀` | 받아들임 |

  F14와 E8은 use의 후보가 하나라서 signature를 바로 기록한다. Example 1(l.332)의 결과(`u16`)도 그대로다.
- 후보가 둘 이상인 use는 후보 집합만 기록한다. 순환 밖에서는 인자의 동치류를 먼저 풀고 rank로 고른다. 의존 그래프는 constraint graph에서 정해지니 순서와 무관하다.
- 하한끼리만 도는 순환의 동치류들은 모든 해에서 같다. ≤는 ⊔가 정의하는 순서(l.114)라 반대칭이다: `S ≤ T`, `T ≤ S`이면 `S = S ⊔ T = T`. 그래서 합쳐도 해를 지우지 않는다. 합치지 않으면 각자의 바깥 하한에서 시작한 한 걸음이 `A ≠ B`일 때 거부하는데, σ는 `A ⊔ B`로 존재한다.
- 순환 안에서, SCC 밖에서 온 하한의 ⊔ `x₀`에서 한 걸음 간 결과가 해이면 `x₀ = σ`다. SCC 밖의 하한은 모든 해에서 하한이라 모든 해 `x* ≥ x₀`이고, `x₀`가 해이므로 최소다. overload 선택의 단조성에 기대지 않는다. 받아들인 값이 정의상 SCC 밖 하한의 ⊔이라, 성질의 국소 검증 두 가지로 확인된다.
- 한 걸음이 거부하는 것은 x가 자라야 하는 순환(`x ≥ A`, `x ≥ f(x)`, `f(A) > A`)이다. 기존 re-fire도 이것을 거부했다. 하한끼리만 도는 순환은 합치므로, 기존보다 받아들이는 범위가 넓다.
- 값이 바뀌지 않을 때까지 반복하는 해법은 택하지 않았다. 최소해임을 보이려면 overload 선택의 단조성이 필요한데 근거가 없고, `x ≥ Option⟨x⟩`처럼 끝나지 않는 경우에 guard가 따로 필요하며, 동치류마다의 검증이 깨진다. 단조성이 증명되면 넓힐 수 있다. 한 걸음이 받아들이는 값에서는 반복도 첫 걸음에서 멈추므로 둘은 맞는다.
- 후보가 둘 이상인 use를 기존 Lifecycle이 순서에 기대어 푼 곳은, 인자가 σ로 정해졌다면, D/R 나눔이 하나로 정해지지 않는 경우(E9)뿐이다.
- 정방향과 역방향은 W1이 가른다. `push(1u8)`의 인자는 caller의 미지수 `T`로 들어가니 경계가 없어 meeting, 곧 정방향이다. `fill(&mut buf)`와 `consume(buf)`는 선언이 parameter 타입을 주니 경계, 곧 역방향이다.
- 역방향은 Fallback이 `!`를 주던 자리를 대신한다. `!`와 literal은 아무것도 정하지 않으니(RFC types §4), 거기서 경계가 coercion 없이 넘어가는 타입을 고른다. 정방향이 정한 것은 덮어쓰지 않으므로 순서와 무관하다.
- 해법 3의 거부와 해법 4의 coercion은 다르다. 3은 정해지지 않은 part를 두고 경계끼리 다투는 경우라 고를 근거가 없다. 4는 타입이 이미 하나로 정해졌고 값만 맞지 않는 경우라, 그 값의 자리에서 바꾼다.
- 증인 (`buf = Vec::new()`, 아래 함수들의 parameter는 선언이 준다):

  | 코드 | `?t` | coercion |
  |---|---|---|
  | `buf.push(1u8)` | `u8` (정방향) | 없음 |
  | `fill(&mut buf)` (`&mut Vec⟨u8⟩`) | `u8` (역방향) | 없음 |
  | `buf.push(1u16); consume(buf)` (`Vec⟨u8⟩`) | `u16` (정방향) | `consume`에서 `Vec⟨u16⟩ → Vec⟨u8⟩` |
  | `let buf = make16(); show16(&buf); consume(buf)` | `u16` (정방향) | `consume`에서 `Vec⟨u16⟩ → Vec⟨u8⟩` |
  | `Vec⟨u16⟩`인 `buf`에 `push(1u8)` | `u16` | `push`의 인자에서 `u8 → u16` |

### 열린 것

1. ~~순환 안의 해법.~~ 하한끼리만 도는 순환은 합치고, use나 constructor를 거치는 순환은 한 걸음으로 정함(해법 2).
2. ~~invariant 규칙.~~ "서로 unify되지 않는 두 등식은 거부"(해법 1), "하한은 `τ ≤ α`, 등식이 `α = U`를 주면 `τ ≤ U`"(해법 2)로 정함.
3. semi callee의 C를 기록하는 방법과, 재귀 group에서 그 기록이 끝나는지(Example 2의 `nest`).
4. LitVar의 default(5절 7)를 끝의 join에서 정하는 기준.
5. ~~경계의 invariant part.~~ 역방향(해법 3)으로 정함. `fill(&mut buf)`의 `?t`는 정방향이 아무것도 정하지 않아 경계가 `u8`로 정한다. conversion의 유무를 part마다 따질 필요가 없다.
6. 다른 constructor끼리의 ⊔(l.118, cast tree의 nca)가 part를 보지 않고 정해지는지. 위에서 아래로 고르려면 필요하다.
