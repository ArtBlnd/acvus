# spec 재구성 장부 (working note)

옛 spec `transparency-constraint-type-system.md.old`(393줄, 고정)의 내용이 있는 줄마다, 새 spec에서 어떻게 처리했는지 적는다. KR `.md.old`는 같은 줄 번호로 1:1 대응하므로 이 장부가 함께 다룬다. 모든 행의 처리가 채워져야 재구성이 끝난다.

처리: **가져옴**(그대로), **고침**(고쳐 씀), **합침**(다른 줄과 합침), **대체**(새 정의로 대체), **버림**(이유를 적음).


## old l.1 Transparency-constraint type system

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|

## old l.3 Transparency

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 5 | An unknown of a run (Lifecycle) is transparent to the run: every constraint of t… | 대체 | KR 풀기/기록(조건 없이 즉시 기록), 기다림, 값. 보류되는 제약이 없어 원리가 구조로 지켜진다 |
| 7 | 1. Every type has one writer: the body whose text makes its values; a declaratio… | 버림 | writer는 보임에서는 graph의 노드로, 경계에서는 run의 미지수로 대체되어 쓰이는 곳이 없다 |
| 9 | A body sees each writer at one of three levels: | 고침 | KR Transparency. writer 대신 use의 대상 |
| 11 | \| Level \| Writer \| What the body sees \| The types that writer writes \| | 고침 | KR Transparency 표(use의 대상, body가 보는 것, 단계) |
| 13 | \| Transparent \| the body itself \| everything \| inferred \| | 고침 | KR Transparency 표 |
| 14 | \| Semi-transparent \| a lambda or an acvus fn it calls \| the signature of its … | 고침 | KR Transparency 표. 텍스트를 use마다 따로 |
| 15 | \| Opaque \| an extern declaration, the host \| its signature \| given \| | 고침 | KR Transparency 표. extern fn과 context의 선언 |
| 17 | 2. Where a value passes into a type, no boundary lies when that type is an unkno… | 고침 | KR 풀기/만남과 경계. run 대신 binding group의 미지수 |
| 18 | 3. An object, enum, tuple, array, option or result type is a structural type: it… | 일부 고침 | KR 타입(structural type 문장). 받은 값의 타입은 풀기 절에서 |
| 19 | 4. A struct's fields are named only inside its impl blocks. | 고침 | KR Transparency(struct의 field 이름과 값) |

## old l.21 Lifecycle

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 23 | Unify, Instantiate, Join and Coercion are applied in this order, round after rou… | 대체 | KR 풀기/Fixpoint와 fallback. round와 단계의 순서 대신 fixpoint, fallback은 전체에서 한 번 |
| 25 | A run of the Lifecycle is the Lifecycle applied once to one set of constraints: … | 대체 | KR 풀기/Binding group. run은 RFC의 run(실행)과 겹쳐 binding group으로, instance의 run은 use가 C를 기록하는 것으로 |
| 27 | A target that meets its step's condition is solved in that step. A target whose … | 고침 | KR 풀기/기다림, 값(기다리는 것이 모두 풀리면 풀린다) |
| 29 | \| Symbol \| Definition \| | 버림 | 표의 머리 줄. 행마다 l.31–39에서 처리 |
| 31 | \| `α ≐ τ` \| α is unified with τ. \| | 버림 | 기호 없이 "unify한다"로 씀(KR 동치류/Unify) |
| 32 | \| `lb(α)` \| α's lower bounds other than α itself. \| | 고침 | KR 풀기/기록, 값. 기호 없이 "하한" |
| 33 | \| `κ(τ)` \| The kind τ's class carries, its constructor, members and parts; it … | 고침 | KR 동치류/Kind. 기호 `κ(τ)`는 쓰지 않음 |
| 34 | \| `dec(τ)` \| τ has no free type variable: `FV(τ) = ∅`. \| | 버림 | 기록과 결정이 정해졌는지를 조건으로 삼지 않아 쓰이는 곳이 없다(KR 풀기/기록) |
| 35 | \| `lit(τ)` \| τ has a free type variable, and every one of them is a LitVar. \| | 버림 | 같은 이유(KR 풀기/기록) |
| 36 | \| `args(u)`, `caps(u)`, `res(u)` \| A use u's arguments, captures and result. \… | 버림 | 기호 없이 "인자", "capture", "결과"로 씀 |
| 37 | \| `callee(u)` \| The type of the function value u calls; none for a fn called b… | 버림 | 기호 없이 "함수 값의 호출"로 씀(KR Instance) |
| 38 | \| `x ⇢ y` \| x waits on y; `⇢⁺` is its transitive closure. \| | 고침 | KR 풀기/기다림. 기호 없이 표로 |
| 39 | \| `σ` \| The pointwise least solution in ≤ of the run's constraints, recorded a… | 대체 | KR 풀기/값, Fixpoint와 fallback. 해는 값의 규칙과 fixpoint가 정한다 |

## old l.41 Unify

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 43 | A type variable α whose only lower bound is τ is unified with τ: α = τ. | 대체 | KR 풀기/값. 하한이 하나인 join |
| 46 | lb(α) = {τ} | 대체 | KR 풀기/값 |
| 47 | ⟹  α ≐ τ | 대체 | KR 풀기/값 |
| 50 | A type is a finite term, and its primitives nest: a term's parts, a projection's… | 일부 고침 | KR 타입 첫 문장, KR 동치류/Unify(재귀 unify). projection·pattern은 남음 |
| 52 | A type variable is one type, represented by an equivalence class kept by union-f… | 고침 | KR 동치류. 동치류와 kind를 나눠 표로 |
| 54 | \| Primitive \| Notation \| Unify rule \| Description \| | 버림 | 표의 머리 줄. 행마다 l.56–69에서 처리 |
| 56 | \| `Var` \| `?0` \| `?0 unify ?1`: one variable, their kinds conjoined. `?0 unif… | 고침 | KR 동치류/Unify 문단(합침, conjunction, occurs check). 표의 행은 지움(리뷰 F23) |
| 57 | \| `Ref(m, t)` \| `&?0`, `&mut ?0` \| `&ₘ?0 unify &ₙ?1`: `m = n`. \| A reference… | 합침 | KR 타입 "같다" 문단. mutability는 constructor에 들어감 |
| 58 | \| `Deref(t)` \| `*?0` \| `*?0 unify τ`: `?0` takes the kind "a reference to `τ`… | 고침 | KR 동치류/Kind(모양), KR 동치류/Projection(규칙과 예) |
| 59 | \| `Field(t, f)` \| `?0.f` \| `?0.f unify τ`: `?0` takes the kind "has field `f:… | 고침 | KR 동치류/Kind(모양), KR 동치류/Projection(규칙과 예) |
| 60 | \| `Part(t, i)` \| `?0.i` \| `?0.i unify τ`: `?0` takes the kind "has part `i: τ… | 고침 | KR 동치류/Kind(모양), KR 동치류/Projection. primitive 이름은 `Component`(리뷰 F25) |
| 61 | \| `Elem(t)` \| `?0[]` \| `?0[] unify τ`: `?0` takes the kind "indexes `τ`". \| … | 고침 | KR 동치류/Kind(모양), KR 동치류/Projection(규칙과 예) |
| 62 | \| `Payload(t, V)` \| `?0 as V` \| `?0 as V unify τ`: `?0` takes the kind "has v… | 고침 | KR 동치류/Kind(모양), KR 동치류/Projection(규칙과 예) |
| 63 | \| `Test(t, π)` \| `?0 is π` \| A binding takes `?0`'s type, by reference as Bin… | 고침 | KR Place(pattern 표). literal은 등식이 아니라 그 자리의 타입으로 넘어가는 값 |
| 64 | \| `Match(t, π̄)` \| `match ?0 { π̄ }` \| A `Test` of `?0` for each arm. \| Seve… | 가져옴 | KR Place(match 문장) |
| 65 | \| `LitVar` \| `?N: int`, `?N: float` \| `?N unify τ`: where `τ` is of `?N`'s fa… | 고침 | KR 동치류/타입 변수(`LitVar` 행), KR 동치류/Unify(한 동치류에 함께 있는 것)(리뷰 F24) |
| 66 | \| `Tuple`, `Array`, `Option`, `Result`, `Fn` \| `(?0, ?1)`, `[?0; n]`, `Option<… | 고침 | KR 타입 표, "같다" 문단 |
| 67 | \| `Object`, `Enum` \| `{f: ?0}`, `E{V(?0)}` \| Same constructor and members. \|… | 고침 | KR 타입 표(Member 열), "같다" 문단 |
| 68 | \| `Extern` \| `N⟨?0, ι⟩` \| Same `N`, and the identity equal. \| An extern-defi… | 고침 | KR 타입 표, "같다" 문단 |
| 69 | \| `Bottom` \| `!` \| `! unify τ`: `τ` is `!`. \| \| | 고침 | KR 타입 표(Bottom 행). `! unify τ`는 KR 동치류/Unify의 `?0 unify τ`에 합침 |

## old l.71 Instantiate

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 73 | Instantiate makes the instances of every instantiable callable: a lambda, and an… | 고침 | KR Instance |
| 75 | 1. Each instantiable callable, a group of acvus fns that call one another togeth… | 고침 | KR 풀기/Binding group. binding group 안에서는 fallback 없이 fixpoint까지 풀고, 남은 것이 C |
| 76 | 2. A use is instantiated once the types of its arguments and of all the callable… | 대체 | KR 풀기/Binding group. 인자가 정해지기를 기다리지 않고 use가 C를 기록하며, type argument는 use 쪽의 미지수 |
| 77 | 3. A use is instantiated at its caller's call site, a call between acvus fns tha… | 고침 | KR Instance 표. call site에서 만듦은 그대로, use가 놓이거나 그것을 감싸는 binding group의 호출은 instance를 만들지 않음 |
| 78 | - where its type arguments are those of an existing instance of the same type sc… | 버림 | use마다 C를 새로 기록하므로 같은 instance를 다시 쓰는 것은 구현의 일이며, 그때도 identity는 새로 만든다(KR Identity) |
| 79 | - otherwise, where its type arguments homeomorphically embed those of an instanc… | 대체 | KR Instance(함수 값 행). 재귀의 경로가 새 instance를 만들지 않으므로 embedding 검사가 필요 없다 |
| 80 | - otherwise, a new instance is made. | 고침 | KR Instance 표(그 밖의 free fn·method 행, 함수 값 행) |
| 82 | Type arguments are compared with identities that differ only by name taken as eq… | 버림 | 같은 이유(l.78) |
| 83 | 4. A use of a function value is instantiated as each lambda its Values fact name… | 고침 | KR Instance(함수 값 행). "그 값의 Values가 가리키는 lambda"(KR Facts) |
| 86 | overload resolution: | 대체 | KR Overload resolution, 풀기/기다림(use는 후보들이 갈리는 자리의 동치류를 기다림) |
| 87 | ∀a ∈ args(u). dec(a) ∨ lit(a) | 대체 | KR Overload resolution. `dec`·`lit` 대신 아무것도 정하지 않는 자리(그 use가 주는 관계를 뺀 자리, `!`, LitVar) |
| 88 | ⟹  a candidate is chosen (Overload resolution) | 대체 | 같은 곳 |
| 89 | instantiable callee: | 버림 | instance는 인자가 정해지기를 기다리지 않는다(KR 풀기/Binding group) |
| 90 | ∀a ∈ args(u) ∪ caps(u) ∪ callee(u). dec(a) | 버림 | 같은 이유 |
| 91 | ⟹  u is instantiated (3) | 버림 | 같은 이유 |
| 92 | waits: | 고침 | KR 풀기/기다림 |
| 93 | u ⇢ a          (a failing the condition above) | 고침 | KR 풀기/기다림(use는 고르는 데 읽는 것을 기다림) |
| 94 | res(u) ⇢ u | 고침 | KR 풀기/기다림(동치류 행, 결정 문단). 자기를 결과로 주는 use는 값의 출처이고 동치류는 그것을 기다린다. 순환은 use가 자기에게서 오는 출처를 읽지 않는 것으로 끊긴다 |

## old l.97 Join

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 99 | Where no boundary lies (Transparency 2), the types that meet are joined. | 고침 | KR 풀기/만남과 경계, 값 |
| 101 | A type variable α is among its own lower bounds: α = α ⊔ τ₁ ⊔ … ⊔ τₙ. Join decid… | 고침 | KR 풀기/기다림(자기 부등식), 값 |
| 104 | D(α) = { τ ∈ lb(α) \| τ ⇢⁺ α } | 고침 | KR 풀기/기다림. 기다림을 따라 자기에게 돌아오는 하한 |
| 105 | R(α) = lb(α) ∖ D(α) | 고침 | KR 풀기/값. 자기에게 돌아오지 않는 하한 |
| 106 | \|lb(α)\| ≥ 2  ∧  R(α) ≠ ∅  ∧  ∀τ ∈ R(α). dec(τ) ∨ lit(τ) | 고침 | KR 풀기/값. 하한의 수와 `dec`·`lit` 대신 기다리는 것이 모두 풀렸는가 |
| 107 | ⟹  α ≐ ⊔R(α) | 고침 | KR 풀기/값(등식이 없음 행). 등식이 있으면 그 타입 |
| 108 | waits: | 고침 | KR 풀기/기다림 |
| 109 | α ⇢ τ          (τ ∈ R(α), ¬dec(τ), ¬lit(τ)) | 고침 | KR 풀기/기다림(동치류 행) |
| 112 | Join takes two types of one constructor with the union of their members, joining… | 합침 | KR 타입/Join의 nca 행. 같은 노드는 공통 조상이 자기 자신(2차 리뷰 D6) |
| 114 | The join of two types of different constructors, and the order `≤` the join defi… | 고침 | KR 타입/Join 표 위 문장. "두 타입의 join과"(2차 리뷰 D6) |
| 116 | \| Definition \| Description \| | 가져옴 | KR 타입/Join |
| 118 | \| `S ⊔ T = nca(S, T)` \| Along the cast tree, the nearest common ancestor. \| | 고침 | KR 타입/Join의 nca 행. 공통 조상 노드까지 올린 뒤 member 합집합, covariant part join(리뷰 F10, 2차 리뷰 D6) |
| 119 | \| `&mut T ⊔ &T = &T` \| Of `&mut T` and `&T`, `&T`. \| | 고침 | KR 타입/Join. `&mut S ⊔ &T = &S ⊔ &T`로 넓힘(리뷰 F7) |
| 120 | \| `! ⊔ T = T` \| Of `!` and `T`, `T`. \| | 가져옴 | KR 타입/Join |
| 121 | \| `?N ⊔ τ = τ` \| A LitVar is unified with a `τ` of its family that holds every… | 고침 | KR 동치류/LitVar의 join. "`?N`과 `τ`를 unify한다" |
| 122 | \| `?N ⊔ ?M = ?(N ∪ M)` \| LitVars of one family are one. \| | 합침 | KR 동치류/LitVar의 join 첫 행과 Unify의 "LitVar `?N`과 `?M`" 규칙으로 덮임(2차 리뷰 D2) |

## old l.124 Coercion

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 126 | Where a boundary lies (Transparency 2), the value is coerced to the type it pass… | 고침 | KR 타입/Coercion 첫 문장(coercion이 무엇인가), KR 풀기/만남과 경계, 값(어디서 일어나는가) |
| 128 | \| Definition \| Description \| | 가져옴 | KR 타입/Coercion. "함수" 열을 더함 |
| 130 | \| `S ⇝ T` for a declared cast `S → T` \| A declared cast. A declared cast goes … | 고침 | KR 타입/Cast tree(정의, 노드, tree는 요구이고 어기면 거부. extern container의 view도 선언된 cast), KR 타입/Coercion(행) |
| 131 | \| `&mut T ⇝ &T` \| `&mut T` to `&T`. \| | 가져옴 | KR 타입/Coercion |
| 132 | \| `{f̄, ḡ} ⇝ {f̄}` \| An object to one with fewer of its fields. \| | 가져옴 | KR 타입/Coercion |
| 133 | \| `E{Ā} ⇝ E{Ā, B̄}` \| An enum to one with more variants. \| | 가져옴 | KR 타입/Coercion |
| 134 | \| `F(T̄) ⇝ F(Ū)` where `Tᵢ ⇝ Uᵢ` at each covariant part \| At each covariant pa… | 가져옴 | KR 타입/Coercion |
| 135 | \| `T⟨φ⟩ ⇝ T⟨ψ⟩` where `φ ⊑ ψ` \| A type to one with higher facts. \| | 버림 | fact는 coercion이 아니라 함수에 붙는 attribute(KR Facts). 만나는 함수 값의 fact는 join된다 |
| 136 | \| `! ⇝ U` \| `!` to any type. \| | 가져옴 | KR 타입/Coercion |
| 137 | \| `T ⇝ U` by a declared conversion `T → U` \| Where none of these reaches the t… | 고침 | KR 타입/Coercion. "위의 행들의 조합으로"(리뷰 F19) |
| 140 | at a boundary where a value v : T passes into U: | 고침 | KR 풀기/기록(경계는 `T ⇝ U`), 값(constructor를 받을 때 part로 쪼갬) |
| 141 | lit(T) ∨ lit(U) | 고침 | KR 풀기/값. `?N ⇝ U`이면 unify |
| 142 | ⟹  T ≐ U | 고침 | 같은 곳 |
| 143 | dec(T) ∧ dec(U) | 고침 | KR 풀기/값(conversion 문단). 양쪽이 정해지기를 기다리지 않고, 동치류가 풀린 뒤 닿지 않는 값만 conversion |
| 144 | ⟹  T ⇝ U       (by the table above; refused where no row reaches U) | 고침 | 같은 곳 |
| 145 | waits: | 버림 | 경계는 기다리지 않고 기록된다(KR 풀기/기록) |
| 146 | v ⇢ T, v ⇢ U   (where undecided) | 버림 | 같은 이유 |

## old l.149 Fallback

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 151 | 1. A type variable with no kind is the bottom type, `!`, where the least solutio… | 고침 | KR 풀기/Fixpoint와 fallback. 전체 fixpoint에서 아무것도 기다리지 않는 동치류에 한 번 |
| 152 | 2. A LitVar takes its default: `i32` for an integer, `f64` for a float. | 고침 | KR 풀기/Fixpoint와 fallback. 정수 `i32`, float `f64`(`.old`를 따름. RFC types §5는 `i64`) |
| 153 | 3. A function's effect is opaque, and its flows are every flow. | 합침 | KR Facts. source가 정하지 않은 fact는 top |
| 156 | when no round changes anything, to each x with ¬∃y. x ⇢ y: | 고침 | KR 풀기/Fixpoint와 fallback. 모든 binding group의 fixpoint 뒤 한 번, C의 원본은 빼고 |
| 157 | κ(α) absent ∧ ¬lit(α) ∧ σ(α) = ! | 대체 | 같은 곳. `σ` 조건 대신 아무것도 기다리지 않음 |
| 158 | ⟹  α ≐ ! | 가져옴 | 같은 곳 |
| 159 | lit(?N) | 고침 | 같은 곳 |
| 160 | ⟹  ?N ≐ i32 for an integer, f64 for a float | 가져옴 | 같은 곳 |
| 161 | an effect not decided | 합침 | KR Facts. source가 정하지 않은 fact는 top |
| 162 | ⟹  opaque, every flow | 합침 | 같은 곳 |

## old l.165 Overload resolution

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 167 | A call resolves over its function's components: its name, its receiver, its argu… | 가져옴 | KR Overload resolution(후보, receiver, 차례 표의 순서) |

## old l.169 Name

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 171 | The candidates are the declarations the name reaches. A structural type answers … | 고침 | KR Overload resolution(후보). structural type의 답은 그 function을 각 part에서 부르는 use, 조건은 types §1 |

## old l.173 Receiver

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 175 | A receiver is an argument tried as `T`, then `&T`, then `&mut T`. | 고침 | KR Overload resolution(receiver). `&mut`는 쓸 수 있는 place일 때, 처음 후보에 닿는 것(types §6) |

## old l.177 Arguments

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 179 | At each argument, in order, the candidates whose parameter it reaches by the lea… | 고침 | KR Overload resolution(차례 표의 인자 행, 아무것도 정하지 않는 자리, conversion 없음) |

## old l.181 Result

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 183 | After the arguments, the result decides, ranked as an argument toward the type i… | 대체 | KR Overload resolution(차례 표의 결과 행). 후보의 결과 타입이, 그 use가 주는 관계를 빼고 정한 결과가 넘어가는 타입에 rank가 가장 낮은 coercion으로 닿음. 정확히 하나, `!`인 인자는 거부하지 않음 |

## old l.185 Rank

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 187 | A coercion is a composition `reborrow? ; cast* ; structural?` of the Coercion ro… | 고침 | KR 타입/Rank. empty는 rank의 값, `!`와 conversion은 rank 없음, 단계와 cast 수는 part까지(리뷰 F16–F20). fact 문장은 fact 절에서 |
| 189 | `empty` is the identity coercion, named apart from the identity of the Identity … | 버림 | empty를 coercion이 아니라 rank의 값으로 두어, identity와 구분하는 문장이 필요 없다(리뷰 F16) |

## old l.191 Context

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 193 | 1. A context `@x` has a fixed type the host gives. | 고침 | KR Transparency(context 문장). position이 없어 reference도 함수도 담지 않음(context §3, ownership §5) |
| 194 | 2. A body may assign to a context and take its value out. A value taken out is a… | 버림 | 타입이 아니라 RFC context §1과 ownership §4가 맡음 |

## old l.196 Parameter

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 198 | 1. A parameter is never assigned. A body may take its value out, and need not pu… | 버림 | 타입에는 필요 없다. parameter로의 대입은 universal로 넘어가는 값이라 경계일 뿐(KR 풀기/만남과 경계). 값을 꺼내고 다시 넣지 않아도 되는 것은 ownership, input에 대입하지 않는 것은 programs §4 |
| 199 | 2. A parameter's type is universal (∀): the use chooses it. | 가져옴 | KR 풀기/Binding group(universal 행) |
| 200 | 3. A script's or a template's input `$x` is a parameter of its function. | 가져옴 | KR Unit(programs §4) |

## old l.202 Lambda

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 204 | A lambda `\|p̄\| e` has the function type `Fn(P̄ → R; φ)`. | 고침 | KR 타입 표(함수), KR Facts(`φ`는 함수에 붙는 attribute) |

## old l.206 Parameters

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 208 | Each `Pᵢ` is a parameter (Parameter). Where a given function type takes the lamb… | 고침 | KR 풀기/Binding group(universal), KR Instance(경계를 넘어 주어진 함수 타입으로 들어가는 행) |

## old l.210 Result

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 212 | `R` is the lambda's result type, whose lower bounds are the types of its tail, i… | 고침 | KR 풀기/만남과 경계. lambda에 한정하지 않고 모든 body의 결과, template은 `String`으로 정해짐 |

## old l.214 Facts

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 216 | `φ` is the lambda's facts: its effect and flows, from its body; its calls and ho… | 합침 | KR Facts 표의 Source 열 |

## old l.218 Captures

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 220 | The names of enclosing bodies that the lambda uses are its captures; how each is… | 고침 | KR 풀기/Binding group(capture는 universal이 아니라 감싸는 binding group의 동치류), 만남과 경계(이름으로 닿는 capture 쓰기는 만남). capture 방식은 inference 뒤라는 문장은 남음 |

## old l.222 Recursion

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 224 | A use that reaches its own lambda through the values it is given is refused. A f… | 대체 | KR Instance(함수 값 행). 거부 대신 use를 품은 그 lambda의 instance를 씀. free fn이 자기와 같은 binding group의 fn을 부르는 것은 Instance 첫 행 |

## old l.226 Module

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 228 | Each is compiled on its own. | 고침 | KR Compilation graph, 풀기/Binding group. 각 binding group이 혼자 풀리고 use가 C를 기록함 |
| 230 | \| Unit \| Free fns \| Struct \| Result type \| | 고침 | KR Unit 표. 결과 타입 열은 KR 풀기/만남과 경계와 template 문장 |
| 232 | \| Module \| any \| at most one, the module itself, whose impl blocks' methods t… | 고침 | KR Unit 표(Module 행). receiver는 KR Struct |
| 233 | \| Script \| one, the script's text its body \| none \| as its body gives \| | 고침 | KR Unit 표(Script 행). 결과는 KR 풀기/만남과 경계 |
| 234 | \| Template \| one, the template's text its body \| none \| `String` \| | 고침 | KR Unit 표와 template 문장. 결과 타입이 `String`으로 정해진 script |
| 236 | A struct has one argument for each of its fields, in the order it declares them:… | 고침 | KR 타입 표(Struct 행), KR Struct. 값 만들기는 unify 대신 각 field로 넘어가는 값, receiver는 텍스트가 적은 kind |

## old l.238 Place

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 240 | A place is a root, a binding, followed by projections, each one of the Unify tab… | 고침 | KR Place 첫 문장. projection은 KR 동치류/Projection |
| 242 | \| Use \| Notation \| Type \| Description \| | 버림 | 표의 머리 줄. 행마다 l.244–248에서 처리 |
| 244 | \| Read \| `p` \| the place's type \| \| | 가져옴 | KR Place 첫 문장(place의 타입) |
| 245 | \| Lend \| `&ₘ p` \| `&ₘ T`; `&ₘ U` where `T` is `&ₙ U` \| A lend of a reference… | 고침 | KR Place(lend), 풀기/기다림(lend의 결과). mutability는 둘 가운데 약한 쪽, target과는 등식 |
| 246 | \| Write \| `p = v` \| `v`'s type meets the place's type \| A store. \| | 고침 | KR 풀기/만남과 경계. 같은 binding group은 Deref·Elem을 거쳐도 만남, 감싸는 binding group은 이름으로 닿을 때만 만남 |
| 247 | \| Test \| a pattern on `p` \| each part the pattern names is a projection of `p… | 고침 | KR Place(pattern 표) |
| 248 | \| Bind \| a name a pattern binds \| by reference where the path passes referenc… | 고침 | KR Place(pattern 표의 binding 행, auto-deref와 reference 판정), 풀기/기다림 |
| 250 | A projection adds the member it names, a field or a variant, to its operand's ty… | 고침 | KR 풀기/값. kind는 값의 출처이며, member가 이 body에 보일 때만 준다. "run의 미지수" 조건과 Reserved의 고정 member는 가시성으로 대체 |

## old l.252 `?`

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 254 | `e?` reads `e`'s type, `Option` or `Result`. | 고침 | KR `?`. 언어가 주는 두 후보를 가진 use로, 고르는 것은 Overload resolution |

## old l.256 Value

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 258 | The value of `e?` is the `Some` or `Ok` payload, a part of `e`'s type. | 고침 | KR `?`. 고른 후보의 결과 `T` |

## old l.260 Exit

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 262 | The exit, `None` or `Err(x)`, enters the innermost body's result as a `return` d… | 고침 | KR `?`(exit는 가장 안쪽 body의 결과로 넘어가는 값). template의 거부는 KR Unit의 template 문장에서 경계로 따라 나옴 |

## old l.264 Error

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 266 | \| Definition \| Description \| | 가져옴 | KR 타입/Join(error join) |
| 268 | \| `⟦!⟧ = ∅`, `⟦Error{N̄(Ē)}⟧ = {Ē}`, any other `⟦E⟧ = {E}` \| A `Result`'s erro… | 가져옴 | KR 타입/Join(error join) |
| 269 | \| `E₁ ⊔ E₂ = ⟦E₁⟧ ∪ ⟦E₂⟧` at a `Result`'s error part \| Of one element, that ty… | 고침 | KR 타입/Join(error join `⊔ᴱ`). 같은 노드는 먼저 `⊔`, 원소 0개는 `!`, 둘 이상은 이름 있는 타입만(리뷰 F12, F13, F14) |

## old l.271 Reserved

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 273 | 1. `Error`, `Option` and `Result` name the language's enums, and `self` the rece… | 일부 고침 | KR 타입/Join의 "`Error`는 언어의 enum". `Option`과 `Result`는 타입 표, `self`는 남음 |
| 274 | 2. `Option`'s members are `Some` and `None`, and `Result`'s `Ok` and `Err`. | 고침 | KR 타입 표(Member 열) |
| 275 | 3. `Error`'s variants are made only by the error join, and a pattern may name th… | 고침 | KR 타입/Join의 "variant는 이 join으로만". pattern이 이름으로 가리키는 것은 KR 풀기/값의 가시성(member가 보이지 않으므로 검사뿐) |

## old l.277 Variance

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|

## old l.279 Covariant

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 281 | \| Type \| Part \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 283 | \| Object \| each field's type \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 284 | \| Enum \| each payload \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 285 | \| Tuple \| each part \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 286 | \| Array \| the element \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 287 | \| Option \| the payload \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 288 | \| Result \| the ok and the error \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |

## old l.290 Invariant

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 292 | \| Type \| Part \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 294 | \| `&T`, `&mut T` \| the target \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 295 | \| Extern-defined type \| each argument, and its identity \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 296 | \| Struct \| each parameter \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 297 | \| Slice \| the element \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 298 | \| Function \| each parameter, and the result \| | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |
| 300 | An invariant part is equal, its facts included. | 일부 가져옴 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |

## old l.302 Contravariant

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 304 | None. | 합침 | KR 타입/Variance 한 문장으로 일반화(2차 리뷰 D1) |

## old l.306 Identity

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 308 | 1. An identity is a type parameter, of an extern-defined type or of a struct. | 고침 | KR Identity. struct는 transparent해서 identity가 없고 part가 각자 지닐 뿐 |
| 309 | 2. An identity that no parameter's type carries is existential, and a call makes… | 고침 | KR Identity. body가 만든 identity는 C에 남아 instance마다 새로 만들어짐 |
| 310 | 3. A type variable and an existential identity carry the level of the execution … | 고침 | KR Identity. escape와 서로 다른 identity는 거부 대신, 값의 제약으로서 identity가 없는 가장 가까운 공통 조상(RFC types §3) |

## old l.312 Effect and Facts

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 314 | 1. A fact is a join-semilattice with a top, nothing known; where a join is not d… | 고침 | KR Facts. source가 정하지 않은 곳도 top |
| 315 | 2. A type carries its facts beside its parts, and a type with lower facts is a s… | 대체 | KR Facts. fact는 함수에 붙는 attribute이며 part도 member도 아니다. subtype 대신 만나는 함수 값의 fact를 join |
| 316 | 3. A fact is decided once, from its sources. | 가져옴 | KR Facts |
| 317 | 4. A function that becomes a value carries its facts in its type, and a call per… | 가져옴 | KR Facts |
| 318 | 5. Inference chooses by no fact that is decided after it. | 가져옴 | KR Facts |
| 320 | \| Fact \| Carried by \| Order \| Top \| Sources \| | 가져옴 | KR Facts |
| 322 | \| Effect \| a function; an extern-defined type \| pure ⊑ opaque \| opaque \| a … | 가져옴 | KR Facts |
| 323 | \| Flows \| a function \| listed flows ⊑ every flow \| every flow \| the functio… | 가져옴 | KR Facts |
| 324 | \| Calls \| a function \| many ⊑ once \| once \| after inference, how its captur… | 가져옴 | KR Facts |
| 325 | \| Holds \| a function \| nothing ⊑ shared ⊑ exclusive \| exclusive \| after inf… | 가져옴 | KR Facts |
| 326 | \| Values \| a function \| sets of lambdas ⊑ any \| any \| the lambdas written \… | 가져옴 | KR Facts. instance에서는 언제나 lambda들의 집합(extern fn은 함수 값을 주지 않고 context는 함수를 담지 않음), KR Instance가 이 이름을 씀 |
| 327 | \| Mark \| a whole concrete extern-defined type \| marked ⊑ unmarked \| unmarked… | 버림 | typeck는 mark를 다루지 않는다(KR 타입의 mark 문장). unmarked를 marked로 바꾸는 것은 optimizer |
| 328 | \| Declared facts \| a declaration \| what is stated ⊑ nothing stated \| nothing… | 가져옴 | KR Facts |

## old l.330 Examples

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|

## old l.332 1. Accumulation in a loop, with literals

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 335 | let mut x = 0; | 대체 | KR Examples 12 |
| 336 | let mut y = 1u16; | 대체 | KR Examples 12 |
| 337 | while c { | 대체 | KR Examples 12 |
| 338 | x = x + y; | 대체 | KR Examples 12 |
| 339 | y = y + 1; | 대체 | KR Examples 12 |
| 340 | } | 대체 | KR Examples 12 |
| 343 | Outcome: `x`, `y` and both literals are `u16`. | 가져옴 | KR Examples 12 |
| 345 | 1. `lb(x) = {?a, r₁}` and `lb(y) = {u16, r₂}`, where `?a` and `?b` are the LitVa… | 대체 | KR Examples 12 |
| 346 | 2. `x + y` waits on `x`, so `r₁ ⇢⁺ x`: `D(x) = {r₁}` and `R(x) = {?a}`. Likewise… | 대체 | KR Examples 12. D/R 대신 그 use가 주는 관계를 빼고 정함 |
| 347 | 3. Round 1, Join: `y ≐ u16` and `x ≐ ?a`; a LitVar in `R(x)` is not waited on. | 대체 | KR Examples 12 |
| 348 | 4. Round 2, Instantiate: the arguments of `x + y` are a LitVar and `u16`, so ove… | 대체 | KR Examples 12 |
| 349 | 5. Coercion: `x`, whose class is a LitVar, and `?b` are unified with the `u16` p… | 대체 | KR Examples 12 |
| 350 | 6. `r₁` and `r₂` are `u16`, and raise neither `x` nor `y`. | 대체 | KR Examples 12. 돌아오는 하한의 검사 |

## old l.352 2. A struct, and recursion that grows

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 355 | struct Pair { l, r } | 가져옴 | KR Examples 14 |
| 356 | impl Pair { | 가져옴 | KR Examples 14 |
| 357 | fn new(l, r) { Self { l, r } } | 가져옴 | KR Examples 14 |
| 358 | fn swap(self) { Self { l: self.r, r: self.l } } | 가져옴 | KR Examples 14 |
| 359 | fn nest(self, n) { if n == 0 { 0u8 } else { Self { l: self, r: 0u8 }.nest(n - 1)… | 가져옴 | KR Examples 14 |
| 360 | } | 가져옴 | KR Examples 14 |
| 364 | let p = Pair::new(1u8, "s"); | 가져옴 | KR Examples 14 |
| 365 | let q = p.swap(); | 가져옴 | KR Examples 14 |
| 366 | q.nest(3) | 가져옴 | KR Examples 14 |
| 369 | Below, `Pair⟨A, B⟩` is the `Self` whose fields `l` and `r` are of types `A` and … | 가져옴 | KR Examples 14 |
| 371 | Outcome: `p: Pair⟨u8, &str⟩` and `q: Pair⟨&str, u8⟩`; `q.nest(3)` is refused at … | 가져옴 | KR Examples 14 |
| 373 | 1. `new`'s parameters are universal, and `Self { l, r }` unifies the fields' typ… | 대체 | KR Examples 14. unify 대신 각 field로 넘어가는 값, C |
| 374 | 2. `Pair::new(1u8, "s")` has its arguments decided and is instantiated: `Pair⟨u8… | 대체 | KR Examples 14. 인자가 정해지기를 기다리지 않고 C의 사본이 인자와 만남 |
| 375 | 3. `q.nest(3)`: an instantiable callee waits for `dec` arguments, and a LitVar i… | 대체 | KR Examples 14. Fallback 대신 Overload의 LitVar default |
| 376 | 4. Inside that instance, `.nest(n - 1)` is a use at `(Pair⟨Pair⟨&str, u8⟩, u8⟩, … | 대체 | KR Examples 14. homeomorphic embedding 대신, 같은 binding group의 재귀가 receiver 경계에서 닿지 못함 |

## old l.378 3. The scope of an identity

| old | 내용 | 처리 | 새 위치 |
|---|---|---|---|
| 381 | let a = open();                       // open() -> F⟨new⟩ | 대체 | KR Examples 11 |
| 382 | a.at(a.index_of());                   // index_of(&F⟨p⟩) -> Idx⟨p⟩,  at(&F⟨p⟩, I… | 대체 | KR Examples 11 |
| 383 | let b = if c { open() } else { a }; | 대체 | KR Examples 11 |
| 384 | let mut v = Vec::new(); | 대체 | KR Examples 11 |
| 385 | while c { v.push(open()); } | 대체 | KR Examples 11 |
| 388 | Outcome: the first two lines are admitted; `let b = …` and the `push` in the loo… | 대체 | KR Examples 11. 거부 대신 identity가 없는 조상 `G` |
| 390 | 1. The identity `ι₀` the first `open()` makes belongs to the execution the call … | 대체 | KR Examples 11 |
| 391 | 2. `index_of`'s `p` is chosen by the use, so its result is `Idx⟨ι₀⟩`, and `at`'s… | 대체 | KR Examples 11 |
| 392 | 3. The `if`'s result has two lower bounds, `F⟨ι₁⟩` and `F⟨ι₀⟩`, so it is a Join.… | 대체 | KR Examples 11 |
| 393 | 4. The `ι₂` the `open()` in the loop makes belongs to its iteration. `push`'s `T… | 대체 | KR Examples 11 |
