# Transparency-constraint type system (한국어)

## Compilation graph

프로그램 하나는 하나의 compilation graph다. 노드는 unit, extern fn, context이고, 간선은 body(free fn, method, lambda의 텍스트)가 이름으로 가리켜 쓰는 use다.

| 노드 | 종류 | Use |
|---|---|---|
| Unit | module, script, template | free fn과 method를 호출한다 |
| Extern fn | 외부 함수 선언 | 호출한다 |
| Context | 선언된 `@x` | 값을 대입하고 꺼낸다 |

### Unit

| Unit | Free fn | Struct |
|---|---|---|
| Module | 0개 이상 | 많아야 하나. module 자신이며, 그 impl block이 method들을 담는다 |
| Script | 하나(script의 텍스트가 body) | 없음 |
| Template | 하나(template의 텍스트가 body) | 없음 |

template은 body가 그 텍스트를 formatting한 `String`인 script이며, 그 결과 타입은 `String`으로 정해져 있다. `{{ e }}`는 `core::display`의 use다. script와 template의 input `$x`는 그 함수의 parameter다(programs §4).

lambda는 다른 body 안에 놓이며, 그 바깥 body가 lambda를 감싸는 body(enclosing body)다. free fn이나 method의 텍스트 안에 놓인 lambda가 하는 호출은 그 free fn이나 method의 호출이다. 호출을 따라 서로에게 닿는 free fn과 method는 한 binding group을 이루고, lambda 하나는, 그 environment를 빼고(Facts), 그 자체로 binding group이며 감싸는 body의 binding group 안에 놓인다.

## Transparency

body는 use의 대상을 세 단계 중 하나로 본다.

| use의 대상 | body가 보는 것 | 단계 |
|---|---|---|
| 자기 자신 | 자기 텍스트 전부 | Transparent |
| 다른 body (free fn, method, lambda) | 그 텍스트를, use마다 따로 | Semi-transparent |
| extern fn, context | 그 선언 | Opaque |

module의 struct는 그 module의 impl block 안에서만 field를 이름으로 가리키고 값을 만들 수 있다.

context의 타입은 host가 준다. context에 담긴 값은 실행이 끝난 뒤에도 남아 그 타입에 position이 없으므로(context §3, ownership §5), context는 reference도 함수도 담지 않는다.

## 타입

타입은 유한한 term이다. term은 constructor 하나와 그 member, part들로 이루어지며, 각 part는 다시 타입이다.

| Constructor | 표기 | Part | Member |
|---|---|---|---|
| Reference | `&T`, `&mut T` | target | |
| Tuple | `(T₁, …, Tₙ)` | 각 component | |
| Array | `[T; n]` | element | |
| Slice | `[T]` | element | |
| Option | `Option⟨T⟩` | payload | `Some`, `None` |
| Result | `Result⟨T, E⟩` | ok, error | `Ok`, `Err` |
| 함수 | `Fn(P̄ → R)` | 각 parameter, result | |
| Object | `{f̄: T̄}` | 각 field의 타입 | field들 |
| Enum | `E{V̄(T̄)}` | 각 payload | variant들 |
| Struct | `Self⟨T̄⟩` | field마다 타입 파라미터 하나, 선언한 순서대로 | |
| Extern-defined type | `N⟨T̄, ι⟩` | 각 argument, 그리고 identity | |
| Scalar | 정수, float, `bool`, `char` (Rust의 scalar type) | 없음 | |
| String | `String` | 없음 | |
| str | `str` | 없음 | |
| Unit type | `()` | 없음 | |
| Bottom | `!` | 없음 | |

reference는 mutability를 가진다. `&`는 shared이고, `&mut`는 exclusive다.

reference를 바로 target으로 갖는 타입은 없다. reference의 lifetime은 loan 여러 개를 담으며, 텍스트에 적히지 않고 use마다 최소의 것으로 정해진다. 다른 body는 Semi-transparent라서, 그 body가 reference로 하는 일이 use마다 보이기 때문이다. 그래서 두 reference의 lifetime을 나눌 필요가 없고, reference를 거쳐 닿는 것은 reference 하나로 닿는다. 그 lifetime은 두 reference의 loan을 함께 담고, 그 reference는 둘 다 exclusive일 때만 exclusive다(ownership §5). target 안쪽의 part는 이 규칙과 상관없다. 예를 들어 `&N⟨&U⟩`의 argument가 그렇다.

두 타입은 constructor, member, part가 모두 같으면 같다. constructor는 표기에서 member와 part를 뺀 나머지다. reference의 mutability, enum의 이름, struct의 module, extern-defined type의 이름, tuple과 array의 길이가 여기에 들어간다.

object, enum, tuple, array, option, result 타입은 structural type이며, 자기가 지닌 이름으로 정해진다.

extern-defined type의 specialization mark(types §8)는 이 type system이 다루지 않는다. 타입은 mark를 지니지 않으며, unmarked인 값을 marked로 바꾸는 것은 optimizer가 한다.

### Variance

structural type은 member에 대해 covariant다. structural type의 part는 cast 없이 합쳐진다. 그 사이의 관계는 part 하한이며, 그 join과 `≤`는 member의 합집합과 `!`, 그리고 안쪽 structural type의 같은 것만으로 이루어진다(Join, Coercion). 그 밖의 타입의 part는 invariant이고, 그 사이의 관계는 등식이다. contravariant인 것은 없다. cast, reborrow, conversion은 값 전체에서만 일어나며 part에서는 일어나지 않는다.

### Cast tree

선언된 cast는 extern-defined type이나 그것을 가리키는 reference에서 가며, reference인지와 그 mutability를 바꾸지 않는다. extern-defined container의 view impl, 곧 `core::as_slice`, `core::as_slice_mut`, `core::as_str`의 impl은, 그 container를 가리키는 reference에서 slice나 `str`을 가리키는 reference로 가는 선언된 cast다(operators §2, §5). cast tree의 노드는 constructor이고, reference의 노드는 mutability와 target의 constructor다. 선언된 cast들은 tree를 이룬다. 곧 한 노드에서 위로 가는 cast는 많아야 하나이고, 순환이 없으며, `&mut X`에서 `&mut Y`로 가는 cast가 있으면 `&X`에서 위로 가는 cast는 `&Y`로 간다.

### Join

두 타입의 join과, join이 정하는 순서 `≤`(`S ⊔ T = T`이면 `S ≤ T`):

| 정의 | 설명 |
|---|---|
| `S ⊔ T = nca(S, T)` | `S`와 `T`가 공통 조상 노드를 가질 때, 각각 cast로 가장 가까운 공통 조상 노드까지 올린 뒤(같은 노드이면 그대로), member는 합집합으로 한다. structural type의 part는 `⊔ₚ`로 합치고, 그 밖의 part는 같아야 한다. 합칠 수 없으면 join이 없다. |
| `&mut S ⊔ &T = &S ⊔ &T` | `&mut S`는 `&S`로 보고 join한다. |
| `! ⊔ T = T` | `!`와 `T`는 `T`. |
| `S ⊔ₚ T` | part의 join. cast 없이 합친다. `S`와 `T`의 constructor가 같을 때 member는 합집합으로 하고, structural type의 part는 다시 `⊔ₚ`로 합치며, 그 밖의 part는 같아야 한다. `! ⊔ₚ T = T`이고, 그 밖에는 join이 없다. 이것이 정하는 순서가 `≤ₚ`다. |

`Error`는 언어의 enum이며, 그 variant는 `?`로만 만들어진다(`?`). 이름이 없는 error 타입 `E`에서 `?`는 variant 대신 `Error⟨E⟩`를 만든다. `Error⟨E⟩`는 variant가 없고 part 하나 `E`를 가진 `Error`이며, 그 constructor는 variant를 가진 `Error`의 것과 달라 둘은 join되지 않는다.

### Coercion

coercion은 값을 다른 타입의 값으로 바꾸는 것이며, 바꾸는 함수가 호출된다. coercion은 타입을 정하지 않는다. 두 쪽의 타입이 다 정해진 뒤, 값 전체가 넘어가는 자리에서 일어난다(값).

| 정의 | 설명 | 함수 |
|---|---|---|
| 선언된 cast `S → T`마다 `S ⇝ T` | cast tree를 한 걸음 올라간다. | 선언의 함수 |
| `&mut T ⇝ &T` | `&mut T`를 `&T`로. | 언어가 정함 |
| `{f̄, ḡ} ⇝ {f̄}` | object를 그 field 중 일부만 가진 object로. | 언어가 정함 |
| `E{Ā} ⇝ E{Ā, B̄}` | enum을 variant가 더 많은 enum으로. | 언어가 정함 |
| structural type의 part마다 `S ⇝ₚ T` | 그 part를 바꾼다. object와 enum 행에서는 남는 member의 part를 바꾼다. | 언어가 정함 |
| `E ⇝ Error{N̄(Ē)}`, `E`가 `Ē`의 하나 | 그 이름의 variant로 감싼다. | 언어가 정함 |
| `E ⇝ Error⟨E⟩`, `E`가 이름이 없는 타입 | `Error⟨E⟩`로 감싼다. | 언어가 정함 |
| `! ⇝ U` | `!`를 어떤 타입으로든. | 언어가 정함 |
| 선언된 conversion `T → U`로 `T ⇝ U` | 위의 행들의 조합으로 그 타입에 닿지 않을 때. | 선언의 함수 |

### Rank

coercion은 Coercion 표의 행들을 `reborrow? ; cast* ; structural?` 순서로 조합한 것이다. reborrow는 `&mut T ⇝ &T` 행, cast는 선언된 cast 행, structural은 object, enum, structural type의 part, `Error` 행이다. part의 coercion `⇝ₚ`는 object, enum, structural type의 part 행과 `! ⇝ U`만의 조합이다. 그래서 cast, reborrow, `Error`로 감싸기, conversion은 part에서 쓰이지 않는다. `! ⇝ U`와 선언된 conversion은 조합이 아니며, rank가 없다. 주어진 타입으로 가는 조합은 하나뿐이다. coercion이 필요 없으면 rank는 `empty`다. 단계의 순서는 `empty ≺ reborrow ≺ cast ≺ structural`이고, coercion의 rank는 가장 높은 단계, 그다음 cast의 수, 그다음 거치는 낮은 단계들 순으로 비교한다.

## 동치류

타입 변수는 하나의 타입이며, union-find의 동치류(equivalence class)로 표현된다. 동치류와 kind는 서로 다른 것이다.

| | 무엇인가 | 담는 것 |
|---|---|---|
| 동치류 | 같은 타입들의 모임 | 타입 변수들, 그리고 kind 많아야 하나 |
| kind | 구조 | constructor, member, 그리고 각 part의 동치류 |

모든 타입은 이렇게 쪼개져 들어간다. 예를 들어 `Vec⟨T⟩`는, `T`의 동치류를 part로 하는 kind `Vec(·)`를 지닌 동치류 하나다.

### 타입 변수

| 종류 | 표기 | 설명 |
|---|---|---|
| `Var` | `?0` | 아직 모르는 타입. |
| `LitVar` | `?N: int`, `?N: float` | literal의 타입. `N`은 그 타입이 담는 literal 값들이며, literal `n`은 `?{n}`이다. 아무것도 정하지 않는다. |

### Kind

kind는 constructor 없이 일부만 알려질 수도 있다. 아래에서 `τ`는 그 타입의 동치류다.

| Kind | 뜻 |
|---|---|
| constructor, member, part | 타입 표의 constructor 하나와 그 member, 그리고 각 part의 동치류. |
| `τ`를 가리키는 reference | target이 `τ`다. |
| field `f: τ`를 가짐 | |
| component `i: τ`를 가짐 | |
| `τ`를 index함 | element가 `τ`다. |
| variant `V`를 가지며 payload가 `τ` | |

### Projection

projection은 아직 모르는 타입의 part를 가리킨다. operand의 동치류에 일부만 알려진 kind를 주고, 그 part의 동치류 `τ`를 가리킨다.

| Primitive | 표기 | 예 | operand의 동치류가 받는 kind |
|---|---|---|---|
| `Deref` | `*?0` | `*r` | `τ`를 가리키는 reference |
| `Field` | `?0.f` | `p.f` | field `f: τ`를 가짐 |
| `Component` | `?0.i` | `t.0` | component `i: τ`를 가짐 |
| `Elem` | `?0[]` | `a[i]` | `τ`를 index함 |
| `Payload` | `?0 as V` | `if let Some(x) = o`의 `x` | variant `V`를 가지며 payload가 `τ` |

kind "`τ`를 index함"은 array와 slice에서는 그 element part와 unify되고, extern-defined container에서는 그 view의 signature를 따라 slice의 element와 이어진다(operators §3). view가 없는 타입은 이 kind를 가질 수 없다.

### Unify

두 동치류를 unify하면 하나로 합쳐지고, kind는 둘의 conjunction이 된다. 이름이 같은 part, 곧 같은 field, 같은 위치의 component, target, element, 같은 variant의 payload는 그 동치류끼리 다시 unify된다(congruence).

| 한 동치류에 함께 있는 것 | 규칙 |
|---|---|
| LitVar `?N`과 constructor | 그 constructor는 `?N`의 계열(정수 또는 float)에 속하고 `N`의 값을 모두 담는 것이다. |
| LitVar `?N`과 `?M` | 같은 계열의 둘은 `?(N ∪ M)` 하나가 된다. |

### LitVar의 join

| 정의 | 설명 |
|---|---|
| `?N ⊔ τ = τ` | `?N`과 `τ`를 unify한다. |

## Facts

fact는 함수에 붙는 attribute이며, 타입의 part도 member도 아니다. fact는 top을 가진 join-semilattice이고, top은 "알 수 없음"이다. join이 정의되지 않는 곳과 source가 정하지 않은 곳에서 fact는 top이다. 한 동치류에서 만나는 함수 값들의 fact는 join된다. fact는 자기 source로부터 한 번 정해지며, inference는 자기보다 나중에 정해지는 fact로 고르지 않는다.

| Fact | 지니는 것 | 순서 | Top | Source |
|---|---|---|---|---|
| Effect | 함수, extern-defined type | pure ⊑ opaque | opaque | 함수의 body. extern-defined type은 그 값이 지닌 함수들 |
| Flows | 함수 | 나열된 flow ⊑ every flow | every flow | 함수의 body |
| Calls | 함수 | many ⊑ once | once | inference 뒤, capture를 어떻게 쓰는지 |
| Holds | 함수 | nothing ⊑ shared ⊑ exclusive | exclusive | inference 뒤, capture |
| Values | 함수 | lambda들의 집합 ⊑ any | any | 쓰인 lambda들 |
| Declared facts | 선언 | 적힌 것 ⊑ 아무것도 안 적힘 | 아무것도 안 적힘 | 선언 |

값이 된 함수는 자기 fact를 그 함수 타입에 지니고, 호출은 callee의 effect를 그 호출이 놓인 body에서 수행한다. 선언과 텍스트가 적는 타입과 context가 주는 타입은 함수 값을 담지 않는다. 함수 값은 lambda의 값뿐이며, 선언은 타입 변수로 받은 함수 값을 돌려줄 수 있을 뿐 새로 만들지 않으므로, instance에서 함수 값의 Values는 언제나 lambda들의 집합이다.

lambda의 텍스트가 이름으로 가리키는 감싸는 body의 binding이 그 lambda의 capture이며, 그 타입은 감싸는 binding group의 동치류다(Binding group). 텍스트가 주는 제약은 자기가 관계짓는 동치류에 걸리며, 감싸는 binding group의 free fn이나 method를 부르는 use는 그 parameter와 결과 타입에도, binding은 그 경로의 projection들의 동치류에도 걸린다. 한 제약이 두 동치류에 함께 걸리거나 하나가 다른 하나에 텍스트가 적은 kind의 part이면 둘은 이어지며, capture의 타입에 바로 또는 다른 동치류를 거쳐 이어진 동치류와 그것들에 걸린 제약이 그 lambda의 environment다. environment는 감싸는 binding group의 것이다. 그 동치류는 그 binding group의 동치류이고 텍스트가 적지 않았으면 그 미지수이며, 그 제약은 lambda가 호출되든 않든 그 binding group이 풀기 시작할 때 한 번 기록되고, 그 안에 놓인 lambda는 자기 environment를 그 텍스트의 제약으로서 먼저 기록한다. 나머지, 곧 environment에 들지 않은 parameter와 결과 타입과 제약은 그 lambda의 binding group이며, 이것을 generic part라 한다. 함수 값을 담는 동치류는 parameter와 결과 타입을 part로 지니지 않고 Values로 그 lambda를 가리킨다. 그 값의 use마다 그 lambda의 C가 instance로 새로 만들어진다(Instance).

## Instance

instance는 use 하나에서 다른 body의 타입들을 새로 만든 것이며, 그 use의 자리(call site)에서 만들어진다(let polymorphism).

| use | instance |
|---|---|
| use가 놓인 binding group이나 그것을 감싸는 binding group의 free fn이나 method를 호출함 | 만들지 않는다. 그 body 자신의 타입을 쓴다. |
| 그 밖의 free fn이나 method를 호출함 | 하나 만든다. |
| 함수 값을 호출함 | 그 함수 값을 읽어(의존), 그 Values가 가리키는 lambda마다 하나 만든다. 각 인자는 그 lambda의 generic part에 든 parameter의 사본이나 environment에 든 parameter로 넘어가고, generic part에 든 결과 타입의 사본이나 environment의 결과 타입이 호출 결과의 하한이 된다. |
| 함수 값이 경계를 넘어 주어진 함수 타입으로 들어감 | 그 경계를 쪼갤 때(값), 그 값의 Values가 가리키는 lambda마다 하나 만든다. 그 함수 타입이 그 lambda를 부르는 것처럼, 그 함수 타입의 각 parameter는 그 lambda의 generic part에 든 parameter의 사본이나 environment에 든 parameter로 넘어가고, generic part에 든 결과 타입의 사본이나 environment의 결과 타입은 그 함수 타입의 result로 넘어간다(만남과 경계). |

embedding이 고른 함수로 시작하는 run은, 그 함수의 instance 하나를 만든다. universal 자리에는 embedding이 그 input에 선언한 타입을 놓고, 그 결과는 embedding이 결과에 선언한 타입으로 넘어간다(programs).

## 풀기

### Binding group

binding group 하나는 constraint graph 하나로 함께 풀린다.

| | 정의 |
|---|---|
| 제약 | 그 binding group의 텍스트가 기록한 제약(안에 놓인 binding group의 텍스트는 빼되, 안에 놓인 lambda의 environment는 넣고)과, 그 안의 use가 기록한 C. |
| universal | parameter의 타입. use가 고른다. capture의 타입은 universal이 아니라 감싸는 binding group의 동치류다. environment에 든 parameter의 타입도 universal이 아니라 감싸는 binding group의 동치류다(Facts). |
| 미지수 | 그 제약에 나오는 동치류 가운데, universal이 아니고, 풀기 시작할 때 정해져 있지 않으며, 감싸는 binding group의 것이 아닌 것. |
| C | 그 binding group이 fixpoint에 닿았을 때 풀리지 않은 동치류와 정해지지 않은 use, lend, projection, binding, 그리고 그것들에 걸린 제약. |
| 순서 | binding group은, 그 C를 기록하는 use가 놓인 binding group보다 먼저 풀린다. |

use는 instance를 만들 때 C에서 그 binding group 자신의 동치류와 use, lend, projection, binding을 새로 만들고, universal 자리에 use의 type argument를 놓아 자기 binding group에 기록한다. 그 사본은 그것이 기록된 binding group에서 정해진다. type argument는 use가 놓인 binding group의 미지수다. 감싸는 binding group의 동치류는 새로 만들지 않고 함께 쓴다. C에서 이미 쪼갠 하한, part 하한, 경계는 사본에서 다시 쪼개지 않는다(값).

### 만남과 경계

값이 어떤 타입으로 넘어가는 자리에서, 그 타입이 그 자리가 놓인 binding group의 미지수이면 만남이고, 아니면 경계다. 이름으로 닿는 binding(그 field와 part 포함)의 타입이 그 자리가 놓인 binding group을 감싸는 binding group의 미지수이면 그것도 만남이다. 감싸는 binding group의 미지수에 reference를 거쳐 닿는 자리는 경계다. body의 결과 타입은 그 body의 미지수이며, tail, `return`, `?`의 exit는 그리로 넘어가는 값이다. template의 결과 타입은 `String`으로 정해져 있어 미지수가 아니다.

### 기록

body의 텍스트가 주는 제약은 주어지는 즉시 동치류에 기록되며, 무엇이 풀렸는지를 조건으로 삼지 않는다.

| 텍스트 | 기록되는 것 |
|---|---|
| 만남 | 하한 `τ ≤ α` |
| 경계 | `T ⇝ U` |
| invariant 자리 | 등식 |
| projection | kind |
| use | 그 타입들이 후보의 signature 가운데 하나와 같다는 제약(Overload resolution). instance를 만들면 그 C다. |
| block expr | 그 자리에서 호출되는 lambda로서 기록되는 것(Block) |

### 의존

| 무엇이 | 무엇에 의존하는가 |
|---|---|
| 동치류 | 풀리지 않은 하한과 part 하한(아무것도 들어오지 않는 동치류는 빼고, 값), 그리고 다음이 아직 기록하지 않은 관계. (가) 자기가 하한이나 part 하한 `τ ≤ α`의 `τ`나 경계 `α ⇝ U`의 `U`의 part이면, `α`가 constructor를 받는 것. 자기가 `?`의 감싸기를 받는 `X`이면, 그 error part가 constructor를 받는 것. (나) 자기가 그 자리에 놓이거나, 그 후보나 pattern이 닿는 깊이 안의 part인 use, lend, projection, binding. 함수 값을 부르는 use의 자리에는, 그 callee의 Values가 그때 가리키는 lambda마다 그 environment에 든 parameter와 결과 타입도 든다(Facts). |
| part의 동치류 | 그것을 part로 가진 동치류들이 모두 constructor를 받는 것 |
| universal | 그것을 고르는 use, 곧 그 body의 instance를 만드는 use. 그 use는 그 binding group 밖에 있다. |
| use | 읽는 자리의 동치류에 관계를 기록하는 use, lend, projection, binding |
| lend, projection, binding | 그 place나 operand의 동치류에 관계를 기록하는 use, lend, projection, binding 가운데, field, component, index, variant의 projection과, 그 동치류가 reference인지를 이미 기록한 use가 아닌 것 |

동치류에 관계를 기록하는 use, lend, projection, binding은, 그 동치류와, 그것에서 동치류끼리의 의존을 따라 닿는 동치류들이 의존하는 use, lend, projection, binding이다. use, lend, projection, binding은 동치류에 의존하지 않는다.

use는 정해지기 전에도 동치류가 reference인지를 기록한다. 그 후보들이 한 동치류에 기록할 하한이나 등식의 타입이 모든 후보에서 reference이면 그 동치류가 reference라는 것을, 어느 후보에서도 reference가 아니면 reference가 아니라는 것을 바로 기록한다. 후보들이 이 점에서 서로 다르면, 그것은 그 use가 정해질 때 기록된다.

예를 들어 `pick({x: i32}) -> {x: i32}`와 `pick(u8) -> u8` 두 선언 아래에서:

```
let mut a;
a.x = 0;
a = pick(a);
```

- `pick`의 결과 `R`은 `A`의 하한이다. 두 후보가 `R`에 기록할 하한 `{x: i32}`와 `u8`은 어느 것도 reference가 아니므로, `R`이 reference가 아니라는 것이 바로 기록된다.
- projection `a.x`는 `A`가 reference인지만 읽으므로 `pick`에 의존하지 않는다. `A`는 reference가 아니고, kind "field `x: τ`를 가짐"은 `A`에 걸린다.
- `pick`은 그 kind에 의존하므로 `A`를 `{x: τ}`로 읽고, `pick({x: i32})`가 남는다.
- 둘째 선언이 `pick(u8) -> &u8`이면 두 후보가 이 점에서 다르므로, `a.x`는 `pick`에 의존하고 `pick`도 `a.x`에 의존한다. 둘은 서로에게 추이적으로 의존하므로 서로로의 의존을 끊으며, `pick`이 읽는 `A`는 아무것도 정하지 않아 그 호출에서 거부된다.

use, lend, projection, binding이 무엇이 되는지는 그것이 읽는 값으로 정해진다. 읽는 값은 그 동치류의 값을, 자기 자신과 자기에게 추이적으로 의존하는 use, lend, projection, binding이 기록하는 관계 없이 남은 관계만으로 정한 것이다. 이때 각 동치류의 값은, 그것이 의존하는 것이 풀렸는지와 상관없이 값의 표, 쪼갬, 채움으로 정하며, 남은 관계가 하나도 없는 동치류는 아무것도 주지 않는다. 선언을 부르는 use는 후보들이 갈리는 자리를 읽고, lend, projection, binding은 그 동치류가 reference인지를 읽는다. reference인지를 읽을 때는 field, component, index, variant의 projection이 기록하는 관계도 기록되지 않았다고 본다(Place). 읽는 값이 아무것도 정하지 않는 자리는 아무것도 정하지 않는다.

의존 그래프는, 그때의 동치류들과 use, lend, projection, binding을 노드로 하고 위 표의 각 행이 주는 의존을 간선으로 하는 그래프다. `Y`가 `X`에 추이적으로 의존한다는 것은 의존 그래프를 따라 `Y`에서 `X`에 닿는다는 것이다. 각 노드는 자기 자신으로의 의존과, 자기에게 추이적으로 의존하는 노드로의 의존을 끊는다. 끊긴 의존이 가리키던 노드가 기록하는 관계는, 그 의존을 끊은 노드가 정해진 뒤에 기록되어 검사된다(값).

끊기지 않은 의존에는 순환이 없다. 순환이 있다면, 그 위의 어느 노드에게든 그 다음 노드는 그 순환을 따라 자기에게 추이적으로 의존하므로, 그 의존은 끊겼을 것이다.

### 값

동치류는 끊기지 않은 의존이 가리키는 것이 모두 풀리거나 정해지면 constructor와 member를 받으며, 이것을 풀린다고 한다. part는 각자 따로 풀린다. 하한과 part 하한끼리만 도는 순환은 먼저 한 동치류로 합친다. 동치류가 기록된 하한, part 하한과 part를 따라 자기 part에 닿으면, 그것을 담는 유한한 타입이 없으므로 그 동치류는 값을 갖지 않으며 더 쪼개지지 않는다.

| | 값 |
|---|---|
| 등식이 있음 | 모든 등식이 주는 타입. |
| 등식이 없음 | 끊긴 의존이 가리키던 하한과 part 하한을 제외한 출처들이 주는 타입의 join, 곧 하한과 use가 주는 타입이 `≤`로, part 하한이 주는 타입이 `≤ₚ`로 닿는 가장 작은 타입. 하한과 part 하한은 자기 타입을, use는 고른 signature의 결과 타입을, kind는 자기가 가리키는 member를 준다. |

kind가 member를 주는 것은 그 member가 이 body에 보일 때뿐이다. object의 field와 enum의 variant는 body의 텍스트가 쓰므로 보이고, `Option`, `Result`, `Error`의 member와 host가 준 타입의 member는 바깥에서 쓰므로 보이지 않는다. 출처가 kind뿐인 동치류는 그 kind가 가리키는 member의 constructor를 받는다. 동치류의 값은 그 동치류가 지닌 kind를 만족하는 타입이다.

동치류가 constructor를 받으면, 그 하한, part 하한, 경계의 두 쪽의 part마다, 값의 part가 넘어가는 자리로서 관계가 기록된다(쪼갬). 그 자리가 만남이면 structural type의 part 사이에는 part 하한이, 그 밖의 part 사이에는 등식이, 경계이면 경계가 기록된다(만남과 경계). 하한이나 경계의 두 쪽의 constructor가 다르면, 값이 선언된 cast들의 parameter로 넘어가고 그 결과가 다시 넘어가는 것으로 보아 같은 규칙으로 잇고, `&mut T`가 `&U`로 넘어가면 target끼리 잇는다. 함수 값이 함수 타입으로 넘어가면, 그 lambda들의 instance로 잇는다(Instance). `?`의 감싸기는 그 감싸는 `Error`의 payload로 잇는다(`?`). `?N ⇝ U`이면 `?N`을 `U`와 unify한다.

동치류가 풀린 뒤, 하한과 use의 결과로 들어오는 값은 Join의 `≤`로, part 하한으로 들어오는 part는 `≤ₚ`로 그 값에 닿으며, 그 단계마다 Coercion 표의 행으로 바뀐다. 다만 object에 더해진 field는 바뀌지 않고, 그 값이 통째로 쓰이기 전에 쓰인다(ownership §4). 경계로 넘어가는 값은 `U`의 값에 coercion으로 닿는다. 어느 쪽이든 그 함수는 그 값이 들어온 자리에서 호출되며, 닿지 않으면 그 자리에서 거부된다.

아무것도 들어오지 않는 동치류, 곧 끊기지 않은 의존이 모두 풀리거나 정해졌는데 출처가 하나도 없는 동치류는, 하한과 part 하한으로서 아무것도 주지 않는다. 그 동치류가 넘어가는 자리, 곧 그것이 `τ`인 하한과 part 하한의 `α`와 그것이 `T`인 경계의 `U`가 모두 값을 가지면, 그 값들이 한 타입일 때 그 동치류는 그 타입을 받고(채움), 서로 다르면 값을 갖지 않는다. 채워진 동치류는 그 자리들에 coercion 없이 닿는다. LitVar만 들어오는 동치류는 쪼갬의 `?N ⇝ U`와 LitVar의 join으로 같은 일을 한다.

### Fixpoint와 fallback

binding group은 더 풀리거나 채워지는 동치류가 없을 때까지 풀린다(fixpoint). binding group 안에서는 fallback하지 않는다.

모든 binding group이 fixpoint에 닿으면, fallback을 한다. 끊기지 않은 의존이 없고 풀리지 않은 동치류는, LitVar이면 그 계열의 default, 곧 정수는 `i32`이고 float는 `f64`가 되며, 그 밖이면 `!`가 된다. C에 남은 동치류에는 fallback하지 않으며, use가 새로 만든 사본에 한다. 그 뒤 다시 fixpoint까지 풀며, fallback할 동치류가 남지 않을 때까지 fallback과 fixpoint를 되풀이한다.

받아들여지는 프로그램은, 선언된 cast들이 Cast tree를 따르고, C에 남은 원본이 아닌 모든 동치류가 값을 갖고, C에 남은 원본이 아닌 모든 use, lend, projection, binding이 정해지며, 들어오고 넘어가는 모든 값이 닿는 프로그램이다. 그 밖의 프로그램은, 정의가 결과를 주지 못한 자리에서 거부된다.

## Place

place는 root인 binding 뒤에 projection들이 이어진 것이며, 그 타입은 마지막 projection이 가리키는 동치류다.

lend `&ₘ p`의 결과는 reference다. lend가 읽는 값에서 place가 `&ₙ U`이면 그 target은 `U`이고 mutability는 `m`과 `n` 가운데 약한 쪽이며, 아니면 target은 place의 동치류다. target과의 관계는 등식이다.

Deref가 아닌 projection은 operand가 reference이면 그 target에 걸린다(auto-deref). 동치류가 reference인지는 field, component, index, variant의 kind를 뺀 출처, 곧 하한, part 하한, 등식, use, Deref의 kind가 정한다.

pattern은 scrutinee place의 projection들이며, 각 자리에서 다음을 기록한다.

| 자리 | 기록되는 것 |
|---|---|
| `(π̄)`, `{f̄: π̄}`, `&π`, `V(π)` | 그 자리의 projection(Component, Field, Deref, Payload) |
| literal | 그 자리의 타입으로 넘어가는 값 |
| binding | 경로가 reference를 지나지 않으면 그 자리를 읽고, 지나면 그 자리의 lend다. mutability는 지난 reference들과 그 자리 자신의 것 가운데 가장 약한 것이다. |
| payload 없는 `V` | `V(_)`와 같다 |
| `_` | 없음 |

match는 arm마다 그 pattern을 scrutinee에 건다.

## Struct

`Self { f̄: ē }`는 constructor가 이 module의 struct이고 part가 그 자리에서 새로 만든 동치류 `T̄`인 값이며, 각 `eᵢ`는 `Tᵢ`로 넘어가는 값이다. struct의 field는 member가 아니라 선언한 순서의 part이므로, field의 kind는 그 part와 unify되고 아무것도 더하지 않는다. receiver `self`, `&self`, `&mut self`의 동치류는 텍스트가 적은 대로 `Self⟨T̄⟩`, `&Self⟨T̄⟩`, `&mut Self⟨T̄⟩`의 kind를 지니며, `T̄`는 universal이다.

`V(e)`는 그 자리에서 새로 만든 동치류 `P`를 `V`의 payload로 가진 값이며, `e`는 `P`로 넘어가는 값이다. `V`가 `Option`이나 `Result`의 member이면 그 constructor는 `Option`이나 `Result`이고 다른 part는 새로 만든 동치류이며, `E::V(e)`이면 이름이 `E`이고 member가 `V` 하나인 enum이다. payload를 쓰지 않은 `V`에서 `P`에는 아무 관계도 없다.

## Overload resolution

선언을 부르는 use의 후보는 이름이 닿는 선언들이다. 연산자, `{{ e }}`, `e?`는 언어가 정한 use이며, 각각 lang item(operators §2), `core::display`, 언어가 주는 두 후보를 후보로 가진다. structural type에서 generic function의 후보는, 그 function을 각 part에서 부르는 것이다(types §1). extern 선언인 후보의 type parameter는 그 use마다 새로 만든, 그 use가 놓인 binding group의 미지수다.

method 호출의 receiver는 첫 인자다. 적힌 그대로, `&`, 그리고 receiver가 쓸 수 있는 place이면 `&mut` 순으로 맞추어, 처음 후보에 닿는 것을 쓴다(types §6).

후보가 여럿인 use는 "그 타입들이 후보의 signature 가운데 하나와 같다"는 제약으로 기록되며, 모든 후보에 공통인 부분은 이 제약에서 바로 성립한다.

후보는 다음 차례로 좁혀지며, 각 차례에서 넘어감이 rank가 가장 낮은 coercion으로 되는 후보만 남는다. 각 자리의 타입은 그 use가 읽는 값이다(의존).

| 차례 | 넘어가는 것 | 넘어가는 타입 |
|---|---|---|
| 인자마다, 차례로 | 인자의 타입 | 후보의 parameter |
| 결과 | 후보의 결과 타입 | 결과가 넘어가는 타입 |

아무것도 정하지 않는 자리는 모든 후보에 닿는다. 그 use가 읽는 값이 정하지 않는 자리, `!`, 그리고 자기 계열 안의 LitVar가 그렇다. 좁힐 때 conversion은 쓰지 않는다(types §7). LitVar의 자리 때문에만 후보가 여럿 남으면, 그 LitVar를 그 계열의 default로 보고 다시 좁힌다. 정확히 하나가 남으면 use는 그 후보로 정해진다. 인자가 `!`인 use는 실행되지 않으므로, 남는 후보와 상관없이 결과가 `!`인 것으로 정해진다(types §6).

## `?`

`e?`는 언어가 주는 두 후보, 곧 `Option⟨T⟩`과 `Result⟨T, E⟩`에서의 후보를 가진 use이며, 그 결과는 `T`다. exit는 `return None`이나 `return Err(x)`와 같다. `x`는 `e`의 error 값을 감싼 값이며, 그 타입은 새로 만든 동치류 `X`다. exit의 값의 타입은 그 자리에서 새로 만든 동치류이며, 가장 안쪽 body의 결과로 넘어가고(만남과 경계), 그 use의 자리다(의존). 감싸기는 `E`가 constructor를 받으면 쪼개진다(값). `E`가 `Error`나 `!`이면 감싸지 않으므로 error 값이 그대로 `X`로 넘어간다. 이름 `N`을 가진 그 밖의 타입(enum, struct, extern-defined type)이면 `X`는 member가 `N` 하나인 `Error`이고, 그 밖이면 `Error⟨..⟩`이며, 어느 쪽이든 그 payload는 새로 만든 동치류이고 error 값은 거기로 넘어간다(Struct). 이 값들은 가장 안쪽 body의 결과에서 만나므로, 그 error part는 `?`마다 variant가 더해진 `Error`가 된다. exit의 다른 part는 새로 만든 동치류이므로(Struct), 그 body가 `Ok`를 돌려주지 않으면 ok part로는 아무것도 들어오지 않고, 그것이 넘어가는 자리가 타입을 주지 않으면(값) fallback으로 `!`가 된다.

## Block

block expr `{ e }`, 곧 그 자체로 식인 block은, parameter가 없고 `e`를 결과로 내는 lambda를 그 자리에서 바로 호출하는 것, 곧 `(|| e)()`로 풀린다(Facts, Instance). if의 branch나 body인 block은 block expr가 아니다. 그래서 `e`에서 capture에 이어진 것은 감싸는 binding group에서 함께 풀리고, 나머지는 자기 binding group에서 먼저 fixpoint까지 풀리며, 거기서 정해진 것은 그대로 쓰이고 정해지지 않아 C에 남은 것은 그 호출의 instance로서 감싸는 binding group에서 정해진다.

## Identity

identity는 extern-defined type의 part이며, 그 값이 가리키는 저장소의 이름이다(ownership §6). struct와 structural type은 identity를 지니지 않고, 그 part가 각자 지닐 뿐이다. 타입 하나는 identity를 많아야 하나 지닌다.

선언의 result에 있는 identity 가운데 어떤 parameter의 타입도 지니지 않은 것은 existential이며, 호출이 그 자리에서 새로 만든다. body가 만든 identity는 그 binding group이 풀어도 C에 남으며, instance마다 새로 만들어진다. 서로 다른 두 identity는 하나가 되지 않는다.

execution은 body나 lambda의 한 번의 호출, 또는 loop의 한 iteration이며, level은 중첩된 execution 사이에서의 깊이다. 동치류는 자기가 생긴 자리의 level을 지니고, 둘이 합쳐지면 얕은 쪽의 level을 지닌다. 호출이 새로 만든 identity는 그 호출이 놓인 execution의 level을 지닌다.

동치류는 서로 다른 identity를 함께 지니지 못하고, 자기보다 깊은 level의 identity도 지니지 못한다. 이것은 동치류의 값에 걸린 제약이므로, 등식이 없는 동치류의 출처들이 그런 identity를 지니면 그 값은 이 제약을 만족하는 가장 작은 타입, 곧 identity를 지니지 않는 가장 가까운 공통 조상이 되고, 각 값은 그 조상으로 coerce된다. part 하한은 cast 없이 닿아야 하므로, part 하한이 그런 identity를 지니면 그 동치류는 값을 갖지 않는다.

## Examples

아래에서 `c`는 `bool`이고, 주석에 signature를 적은 함수는 extern 선언이다.

### 1. 한쪽에만 정보가 있는 if

```
let a = if c { Some(1u8) } else { None };
```

결과: `a`는 `Option⟨u8⟩`이다.

1. 두 branch의 값이 `a`의 동치류 `A`와 만나, `A ≥ Option⟨S⟩`와 `A ≥ Option⟨?k⟩`가 기록된다. `S`는 `Some`이 새로 만든 payload이고 `S ≥ u8`이며, `?k`는 `None`이 새로 만든 payload이고 아무 관계도 없다.
2. 두 하한은 모두 constructor `Option`으로 풀려 있으므로 `A`는 `Option`을 받고, 그 payload `P`에 part 하한 `S ≤ₚ P`와 `?k ≤ₚ P`가 기록된다.
3. `S`는 `u8`이고, `?k`로는 아무것도 들어오지 않아 아무것도 주지 않으므로 `P = u8`이다. `?k`는 넘어가는 자리 `P`의 `u8`로 채워진다.

### 2. 경계가 채우는 part

```
let a = Vec::new();      // new() -> Vec⟨T⟩
fill_all(&mut a);        // fill_all(&mut Vec⟨u8⟩)
```

결과: `a`는 `Vec⟨u8⟩`이다.

1. `new`의 `T`는 이 use에서 새로 만든 미지수 `?t`이고, `A ≥ Vec⟨?t⟩`가 기록된다.
2. `&mut a`의 결과는 target이 `A`인 `&mut`이다. 그 값은 선언이 준 `&mut Vec⟨u8⟩`로 넘어가므로 경계다.
3. 경계를 쪼개면, target 쪽 `Vec⟨u8⟩`은 미지수가 아니므로 경계 `A ⇝ Vec⟨u8⟩`이 기록된다. `A`가 하한으로 `Vec`을 받으면, 하한에서는 `Vec`의 argument가 invariant이므로 등식으로 `A`의 part가 `?t`와 같아지고, 이 경계에서는 `u8`이 미지수가 아니므로 경계 `?t ⇝ u8`이 기록된다.
4. `?t`로는 아무것도 들어오지 않으므로, 넘어가는 자리의 `u8`로 채워진다. `A`는 `Vec⟨u8⟩`이다.

### 3. 경계에 적힌 타입은 값을 정하지 않는다

```
let mut buf = Vec::new();
push(&mut buf, 1u16);    // push(&mut Vec⟨T⟩, T)
consume(buf);            // consume(Vec⟨u8⟩)
```

결과: 거부된다. `buf`는 `Vec⟨u16⟩`이고, `consume(buf)`에서 `Vec⟨u16⟩`이 `Vec⟨u8⟩`에 닿지 않는다. 선언된 conversion `u16 → u8`이 있어도 part에서는 쓰이지 않는다.

1. 2번처럼 `B ≥ Vec⟨?t⟩`이다.
2. `push`의 `T`는 새 미지수 `?T`다. 첫 인자는 경계 `&mut B ⇝ &mut Vec⟨?T⟩`이고, 쪼개면 경계 `B ⇝ Vec⟨?T⟩`가, 다시 쪼개면 `?T`가 미지수이므로 등식 `?t = ?T`가 기록된다. 둘째 인자는 미지수 `?T`로 넘어가므로 만남이다: `?T ≥ u16`.
3. `consume(buf)`는 경계 `B ⇝ Vec⟨u8⟩`이고, 쪼개면 `u8`이 미지수가 아니므로 경계 `?t ⇝ u8`이 기록된다.
4. `?t`, 곧 `?T`로는 `u16`이 들어오므로 그 값은 `u16`이다. 경계에 적힌 `u8`은 값을 정하지 않는다.
5. `B`는 `Vec⟨u16⟩`이다. `consume`의 인자에서 `Vec⟨u16⟩`을 `Vec⟨u8⟩`로 바꾸는 coercion이 없으므로 거부된다.

### 4. 넘어가는 자리가 둘인 빈 part

```
let x = None;
let y = if c { x } else { Some(1u16) };
take(x);                 // take(Option⟨u8⟩)
```

결과: 거부된다. `y`는 `Option⟨u16⟩`이고, `x`의 payload는 넘어가는 자리가 `u16`과 `u8`을 주어 채워지지 않는다.

1. `X ≥ Option⟨?k⟩`, `Y ≥ X`, `Y ≥ Option⟨S⟩`, `S ≥ u16`.
2. `X`가 `Option`을 받으면 그 payload `P`에 part 하한 `?k ≤ₚ P`가 기록된다. `take(x)`는 경계 `X ⇝ Option⟨u8⟩`이고, 쪼개면 `u8`이 미지수가 아니므로 경계 `P ⇝ u8`이 기록된다.
3. `Y`가 `Option`을 받으면 그 payload `Q`에 part 하한 `P ≤ₚ Q`와 `S ≤ₚ Q`가 기록된다. `?k`와 `P`로는 아무것도 들어오지 않아 아무것도 주지 않으므로, `Q`는 `u16`이다.
4. `P`가 넘어가는 자리는 `Q`와 `take`의 `u8`이고, 그 값 `u16`과 `u8`은 한 타입이 아니다. 그래서 `P`는 값을 갖지 않아 거부된다.

### 5. 자기 part에 닿는 하한

```
let mut x = None;
x = Some(x);
```

결과: 거부된다.

1. `X ≥ Option⟨?k⟩`이고, `Some(x)`는 새 payload `S`를 가진 `Option⟨S⟩`이며 `S ≥ X`이므로 `X ≥ Option⟨S⟩`다.
2. `X`가 `Option`을 받으면 그 payload `P`에 part 하한 `?k ≤ₚ P`와 `S ≤ₚ P`가 기록된다.
3. `X`는 part를 따라 `P`에 닿고, `P`는 part 하한을 따라 `S`에, `S`는 하한을 따라 다시 `X`에 닿는다. `X`가 자기 part 안에 다시 나타나므로, 그것을 담는 유한한 타입이 없어 거부된다.

### 6. 이름으로 닿는 capture 쓰기

```
let mut last = None;
let remember = |x| last = Some(x);
remember(1u8);
```

결과: `last`는 `Option⟨u8⟩`이다.

1. `L ≥ Option⟨?k⟩`.
2. `last = Some(x)`는 capture `L`을 이름하고, `Some(x)`의 payload `S`와 `S ≥ X`로 `x`의 타입 `X`가 그것에 이어지므로, 이것들은 lambda의 environment이며 감싸는 binding group의 것이다(Facts). `X`는 universal이 아니라 그 binding group의 미지수다. `last`는 이름으로 닿고 `L`은 그 binding group의 미지수이므로, 쓰기는 만남이다: `L ≥ Option⟨S⟩`, `S ≥ X`. generic part에는 결과의 `()`만 남는다.
3. `remember(1u8)`는 Values가 이 lambda뿐인 함수 값의 호출이라 instance를 만든다. `X`는 environment에 들므로 인자는 `X` 자신과 만난다: `X ≥ u8`. generic part의 C에는 아무것도 남지 않아 사본은 아무것도 더하지 않는다.
4. `L`의 payload `P`에는 part 하한 `?k ≤ₚ P`, `S ≤ₚ P`가 기록된다. `S`의 하한은 `X`뿐이고 `X = u8`이므로 `S = u8`이며, `?k`는 아무것도 주지 않으므로 `P = u8`이다.

### 7. reference를 거친 쓰기

```
let mut x = None;
let r = &mut x;
*r = Some(1u8);
```

결과: `x`는 `Option⟨u8⟩`이다.

1. `X ≥ Option⟨?k⟩`. `X`는 reference가 아니므로 `r`의 타입은 target이 `X`인 `&mut`이다.
2. `*r`의 타입은 `r`의 target `X`다. 쓰기가 같은 binding group에 있고 `X`는 그 미지수이므로 만남이다: `X ≥ Option⟨S⟩`, `S ≥ u8`.
3. `X`의 payload에는 part 하한 `?k`와 `S`가 기록된다. `S`는 `u8`이고 `?k`는 아무것도 주지 않으므로 그 payload는 `u8`이다.

같은 쓰기를 lambda 안에서 capture한 `r`을 거쳐 하면, `X`가 감싸는 binding group의 미지수라도 reference를 거쳐 닿으므로 경계다.

### 8. kind와 가시성

```
let a;
a.x = 0;
```

결과: `a`는 `{x: i32}`다.

1. projection `a.x`가 `A`에 kind "field `x: τ`를 가짐"을 주고, 쓰기는 `τ ≥ ?{0}`이다.
2. `A`의 출처는 kind뿐이다. field는 이 body의 텍스트가 쓰므로 보이고, `A`는 그 member의 constructor인 object를 받아 `{x: τ}`가 된다.
3. `τ`에는 LitVar뿐이라 fallback에서 `i32`가 된다.

```
@cfg.extra = 1;          // @cfg는 host가 준 { name }
```

결과: 거부된다. `@cfg`의 member는 host가 쓰므로 보이지 않아, kind "field `extra`를 가짐"은 member를 더하지 못한다. `{name}`과 그 kind의 conjunction을 가질 타입이 없다.

```
if let Ok(v) = Some(1u8) { }
```

결과: 거부된다. `Option`의 member는 언어가 쓰므로 보이지 않아, kind "variant `Ok`를 가지며 payload가 `τ`"는 member를 더하지 못한다.

### 9. reference를 다시 빌리는 lend

```
let mut v = Vec::new();
let r = &mut v;
let s = &r;
```

결과: `s`는 target이 `v`의 동치류인 `&`다.

1. `r`의 타입은 target이 `V`인 `&mut`이다.
2. `&r`은 place `r`을 빌린다. 그 동치류는 reference `&mut V`이므로 target은 `V`이고, mutability는 shared와 exclusive 가운데 약한 shared다.

### 10. 이름 호출의 재귀

```
fn g(a, b) { g(b, a) }
```

```
g(1u8, "s")
```

결과: `g(1u8, "s")`의 instance 안, 재귀 호출에서 거부된다.

1. `g`의 재귀 호출은 같은 binding group의 호출이라 instance를 만들지 않고 `g` 자신의 타입을 쓴다. parameter의 타입 `A`, `B`는 universal이라 미지수가 아니므로, 인자가 넘어가는 자리는 경계다: `B ⇝ A`, `A ⇝ B`. 둘은 universal에 기대어 `g`의 C에 남는다.
2. `g(1u8, "s")`는 다른 binding group의 호출이라 instance를 만든다. `A'`, `B'`는 새 미지수이고 인자와 만나 `A' = u8`, `B' = &str`이 된다.
3. C의 사본인 경계 `&str ⇝ u8`에는 닿는 조합도 선언된 conversion도 없어 거부된다.

한 binding group 안의 재귀는 새 instance를 만들지 않으므로, 두 parameter의 타입을 바꿔 부르는 재귀는 받아들여지지 않는다.

### 11. identity

```
let a = open();                     // open() -> F⟨new⟩, F에서 identity가 없는 G로 가는 cast가 선언됨
at(&a, index_of(&a));               // index_of(&F⟨p⟩) -> Idx⟨p⟩, at(&F⟨p⟩, Idx⟨p⟩)
let b = if c { open() } else { a };
let mut v = Vec::new();
while c { push(&mut v, open()); }   // push(&mut Vec⟨T⟩, T)
```

결과: 둘째 줄은 identity 하나로 받아들여지고, `b`는 `G`, `v`는 `Vec⟨G⟩`이다. `G`로 가는 cast가 없다면 `let b`와 `push`는 거부된다.

1. 첫 `open()`이 만든 identity `ι₀`는 이 body의 execution의 것이고, `a: F⟨ι₀⟩`는 같은 level이다.
2. `index_of`의 `p`는 이 use의 새 미지수다. `&F⟨ι₀⟩`가 `&F⟨p⟩`로 넘어가는 경계를 쪼개면, identity `p`는 미지수이므로 등식으로 `ι₀`가 되고, 결과는 `Idx⟨ι₀⟩`다. `at`의 `p`도 `ι₀` 하나로 맞는다.
3. `if`의 결과 `B`에는 `F⟨ι₁⟩`과 `F⟨ι₀⟩`가 하한으로 기록된다. `B`는 서로 다른 identity를 함께 지니지 못하므로, 그 값은 identity가 없는 가장 가까운 공통 조상 `G`이고, 두 값은 각각 `G`로 cast된다.
4. loop 안의 `open()`이 만든 `ι₂`는 그 iteration의 것이다. `push`의 `T`는 이 use에서 생기지만, `&mut Vec⟨T⟩`로 넘어가는 경계를 쪼개면 `T`는 미지수이므로 등식으로 `v`의 element와 합쳐지고, 얕은 쪽인 바깥 level을 지닌다. 그 동치류는 더 깊은 `ι₂`를 지니지 못하므로 값은 `G`가 되고, 넣는 값은 `G`로 cast된다.

### 12. literal에서 시작한 누적

```
let mut x = 0;
let mut y = 1u16;
while c {
    x = x + y;
    y = y + 1;
}
```

결과: `x`, `y`와 두 literal 모두 `u16`이다.

1. `X ≥ ?a`, `X ≥ R₁`, `Y ≥ u16`, `Y ≥ R₂`가 기록된다. `?a`, `?b`는 `0`, `1`의 LitVar이고, `R₁`, `R₂`는 `x + y`, `y + 1`의 결과다.
2. `x + y`는 `core::add(&x, &y)`의 use이고, 후보인 `add<T, O>(&T, &T) -> O`의 impl들은 `T`에서 갈린다. `y + 1`도 같다.
3. 두 use는 모두 `&y`의 target `Y`에 관계를 기록하고 `Y`를 읽으므로, 서로에게 추이적으로 의존한다. 그래서 각 use가 읽는 값에는 두 use의 관계가 모두 없다. 그 값에서 `Y`는 텍스트가 준 `u16`이고, `R₁`, `R₂`에는 아무 관계도 없다. 각 lend는 그 place가 reference인지만 읽는데, `add`의 후보들은 어느 것도 `X`, `Y`, `R₁`, `R₂`에 reference를 기록하지 않으므로 두 use에 의존하지 않고, 그 place가 reference가 아니라고 정한다.
4. `y + 1`에서 `Y`의 자리는 `u16`이고 `?b`의 자리는 아무것도 정하지 않으므로 `u16`의 `add`가 남는다. `x + y`에서 `X`의 자리는 LitVar `?a`뿐이라 정수 계열 안에서 아무것도 정하지 않고, `Y`의 자리는 `u16`이므로 `u16`의 `add`가 남는다.
5. 정해진 두 `add`의 경계를 쪼개면, `u16`은 미지수가 아니므로 경계 `X ⇝ u16`, `Y ⇝ u16`, `?b ⇝ u16`이 기록되고, `R₁`, `R₂`는 결과 타입 `u16`을 받는다. `X`는 `?a`와 `R₁`의 join으로 `u16`이 되어 `?a`는 `u16`과 unify되고, `Y`는 `u16`이며, `?b`는 `?N ⇝ U`로 `u16`과 unify된다. 세 경계는 그대로 닿는다.

### 13. 결과로 고르는 overload

```
fill_all(&mut parse(s));     // parse(&str) -> Vec⟨u8⟩와 parse(&str) -> Vec⟨u16⟩ 두 선언, fill_all(&mut Vec⟨u8⟩)
let _ = &mut parse(s);
```

결과: 첫 줄의 `parse`는 `→ Vec⟨u8⟩`이고, 둘째 줄은 거부된다.

1. `parse`의 두 후보가 결과 `R`에 기록할 `Vec⟨u8⟩`과 `Vec⟨u16⟩`은 어느 것도 reference가 아니므로, `R`이 reference가 아니라는 것이 바로 기록된다. 그래서 첫 줄의 lend는 `parse`에 의존하지 않고 target을 `R`로 정한다. 그 reference는 선언이 준 `&mut Vec⟨u8⟩`로 넘어가는 경계이고, 쪼개면 `Vec⟨u8⟩`은 미지수가 아니므로 경계 `R ⇝ Vec⟨u8⟩`이 기록된다.
2. 두 후보는 인자로는 갈리지 않으므로 결과로 좁혀진다. `parse`가 읽는 값에서 `R`로는 아무것도 들어오지 않으므로, `R`은 넘어가는 자리의 `Vec⟨u8⟩`로 채워진다. 거기에 닿는 것은 `→ Vec⟨u8⟩`뿐이다.
3. 둘째 줄에서 `parse`가 읽는 `R`의 값은 아무것도 정하지 않는다. 후보 둘이 남아 그 호출에서 거부된다.

### 14. struct와 커지는 재귀

```
struct Pair { l, r }
impl Pair {
    fn new(l, r) { Self { l, r } }
    fn swap(self) { Self { l: self.r, r: self.l } }
    fn nest(self, n) { if n == 0 { 0u8 } else { Self { l: self, r: 0u8 }.nest(n - 1) } }
}
```

```
let p = Pair::new(1u8, "s");
let q = p.swap();
q.nest(3)
```

아래에서 `Pair⟨A, B⟩`는 field `l`, `r`의 타입이 `A`, `B`인 `Self`다.

결과: `p`는 `Pair⟨u8, &str⟩`, `q`는 `Pair⟨&str, u8⟩`이고, `q.nest(3)`은 그 instance 안의 재귀 호출에서 거부된다.

1. `new`에서 `Self { l, r }`는 part가 새 동치류 `P`, `Q`인 `Pair⟨P, Q⟩`를 만들고, `P ≥ L`, `Q ≥ R`이다. `L`, `R`은 parameter의 타입이라 universal이므로 이것들은 C에 남는다. `Pair::new(1u8, "s")`의 instance에서는 그 사본이 인자와 만나므로 `p`는 `Pair⟨u8, &str⟩`이다.
2. `swap`의 receiver는 `Pair⟨A, B⟩`의 kind를 지니고, `self.r`, `self.l`은 part `B`, `A`다. 그래서 결과는 `P ≥ B`, `Q ≥ A`인 `Pair⟨P, Q⟩`다. `p.swap()`에서 `p`가 receiver로 넘어가는 경계를 쪼개면, `A`, `B`의 사본은 미지수이므로 등식으로 `u8`, `&str`이 되고, `q`는 `Pair⟨&str, u8⟩`이다.
3. `nest`의 재귀 호출은 같은 binding group의 호출이라 instance를 만들지 않는다. 그 receiver `Self { l: self, r: 0u8 }`는 `P ≥ Pair⟨A, B⟩`, `Q ≥ u8`인 `Pair⟨P, Q⟩`다. 이 값이 receiver `Pair⟨A, B⟩`로 넘어가는 자리는 경계이고, 쪼개면 `A`, `B`는 universal이므로 경계 `P ⇝ A`, `Q ⇝ B`가 기록된다. `P`는 하한으로 `Pair⟨A, B⟩`, `Q`는 `u8`이 되며, 두 경계는 universal에 기대므로 C에 남는다.
4. `q.nest(3)`의 instance에서 `A`, `B`의 사본은 `&str`, `u8`이다. `n == 0`과 `n - 1`은 LitVar의 자리 때문에만 후보가 여럿 남으므로, LitVar를 default `i32`로 보고 `i32`로 좁혀진다.
5. C의 사본인 경계 `Pair⟨&str, u8⟩ ⇝ &str`에는 닿는 조합도 선언된 conversion도 없다. 그래서 그 재귀 호출에서 거부된다.