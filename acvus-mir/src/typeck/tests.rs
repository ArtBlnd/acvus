//! Each test states the rule its expected result comes from. Programs are
//! parsed with `acvus-ast`; declarations go through `verify`.

use acvus_ast::{BinOp, ElseBranch, Expr, Script, Stmt, Template};
use acvus_utils::{Astr, Interner, QualifiedRef};

use super::{Capture, Captured, Impl, Iteration, RefusalKind, TryExit, Typeck, check_script, check_template};
use crate::decl::{CastRule, Core, Declarations, ImplSig, Impls, LoopMarkers, RawDeclarations, Scheme, TyVarBound, UserTypeDecl};
use crate::ty::{
    Calls, Concrete, EffectTerm, FlowTerm, Flows, Holds, IntTy, LenTerm, Mutability, ObjectTy, Poly, Ty, TyId, TyKind,
    TypeArg, Types,
};

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
}

/// A checked program and the table its types are in.
struct Checked {
    typeck: Typeck,
    types: Types<Concrete>,
}

impl Checked {
    fn kind(&self, ty: Ty) -> &TyKind<Concrete> {
        self.types.kind(ty)
    }

    fn refusals(&self) -> Vec<&RefusalKind> {
        self.typeck.refusals.iter().map(|refusal| &refusal.kind).collect()
    }

    fn admitted(&self) {
        assert!(self.typeck.refusals.is_empty(), "refused: {:?}", self.typeck.refusals);
    }
}

impl Fixture {
    fn new() -> Self {
        let names = Interner::new();
        let core = Core::of(&names);
        Self {
            names,
            raw: RawDeclarations {
                lang_impls: Vec::new(),
                types: Types::new(),
                functions: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                inputs: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
        }
    }

    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self, width: IntTy) -> TyId<Poly> {
        self.ty(TyKind::Int(width))
    }

    fn var(&mut self, at: u32) -> TyId<Poly> {
        self.ty(TyKind::Var(at))
    }

    fn reference(&mut self, mutability: Mutability, target: TyId<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Ref(mutability, TypeArg::Uniform(target)))
    }

    fn function(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, effect: EffectTerm<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect,
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
        })
    }

    fn pure(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>) -> TyId<Poly> {
        self.function(params, ret, EffectTerm::Known(crate::ty::Effect::pure()))
    }

    fn param_names(&self, count: usize) -> Box<[Astr]> {
        (0..count).map(|at| self.name(&format!("p{at}"))).collect()
    }

    fn arity(&self, ty: TyId<Poly>) -> usize {
        match self.raw.types.kind(ty) {
            TyKind::Fn { params, .. } => params.len(),
            _ => 0,
        }
    }

    /// A plain extern fn `name`, its type variables bounded by `bounds`.
    fn declare(&mut self, name: &str, ty: TyId<Poly>, bounds: &[TyVarBound]) -> QualifiedRef {
        let id = QualifiedRef::root(self.name(name));
        let scheme = Scheme {
            ty,
            param_names: self.param_names(self.arity(ty)),
            bounds: bounds.into(),
            impls: None,
            requires: Box::new([]),
        };
        self.raw.functions.push((id, scheme));
        id
    }

    /// A plain extern fn `name` in `namespace`, one of the declarations a
    /// bare `name` may reach.
    fn overload(&mut self, namespace: &str, name: &str, ty: TyId<Poly>) -> QualifiedRef {
        let id = QualifiedRef::qualified(self.name(namespace), self.name(name));
        let scheme = Scheme {
            ty,
            param_names: self.param_names(self.arity(ty)),
            bounds: Box::new([]),
            impls: None,
            requires: Box::new([]),
        };
        self.raw.functions.push((id, scheme));
        id
    }

    /// An extern `Vec<T>` whose `as_slice` impl views it as `&[T]`
    /// (operators §3, §5).
    fn vec_with_as_slice(&mut self) -> QualifiedRef {
        let (vec_id, at_vec) = self.vec_and_its_view();
        let as_slice = self.raw.core.as_slice;
        self.lang_item(as_slice, &[at_vec]);
        vec_id
    }

    /// An extern `Vec<T>`, and the type of its `as_slice` impl,
    /// `(&Vec<T>) -> &[T]`, not yet given.
    fn vec_and_its_view(&mut self) -> (QualifiedRef, TyId<Poly>) {
        let vec_id = QualifiedRef::root(self.name("Vec"));
        self.raw.user_types.push(UserTypeDecl {
            id: vec_id,
            type_params: Box::new([TyVarBound::Any]),
            specializes: Box::new([false]),
            effect_params: 0,
            identity_params: 0,
            lifetimes: 0,
        });
        let e = self.var(0);
        let vec_e = self.vec_of(vec_id, e);
        let shared_vec_e = self.reference(Mutability::Shared, vec_e);
        let slice_e = self.ty(TyKind::Slice(e));
        let shared_slice_e = self.reference(Mutability::Shared, slice_e);
        (vec_id, self.pure(&[shared_vec_e], shared_slice_e))
    }

    /// `Vec<element>`.
    fn vec_of(&mut self, vec_id: QualifiedRef, element: TyId<Poly>) -> TyId<Poly> {
        self.ty(TyKind::UserDefined {
            id: vec_id,
            type_args: Box::new([TypeArg::Uniform(element)]),
            effect_args: Box::new([]),
            identity_args: Box::new([]),
        })
    }

    /// A lang item's concrete impls; its generic function is the
    /// compiler's (operators §2).
    fn lang_item(&mut self, item: QualifiedRef, impls: &[TyId<Poly>]) {
        let concrete = impls
            .iter()
            .map(|ty| ImplSig {
                ty: *ty,
                param_names: self.param_names(self.arity(*ty)),
                requires: Box::new([]),
                facts: Default::default(),
            })
            .collect();
        self.raw.lang_impls.push((item, Impls { concrete, generic: None }));
    }

    /// Impls of `add<T, O>(&T, &T) -> O`, or of one of its binary
    /// siblings, at the widths given, each `(&w, &w) -> w`.
    fn arithmetic(&mut self, item: QualifiedRef, widths: &[TyKind<Poly>]) {
        let impls: Vec<TyId<Poly>> = widths
            .iter()
            .map(|kind| {
                let w = self.ty(kind.clone());
                let shared = self.reference(Mutability::Shared, w);
                self.pure(&[shared, shared], w)
            })
            .collect();
        self.lang_item(item, &impls);
    }

    /// Impls of `display<T>(&T, out: &mut String)` at `shown`.
    fn display(&mut self, shown: &[TyKind<Poly>]) {
        let string = self.ty(TyKind::String);
        let out = self.reference(Mutability::Mut, string);
        let unit = self.ty(TyKind::Unit);
        let impls: Vec<TyId<Poly>> = shown
            .iter()
            .map(|kind| {
                let w = self.ty(kind.clone());
                let shared = self.reference(Mutability::Shared, w);
                self.pure(&[shared, out], unit)
            })
            .collect();
        let item = self.raw.core.display;
        self.lang_item(item, &impls);
    }

    fn context(&mut self, name: &str, ty: TyId<Poly>) {
        let id = QualifiedRef::root(self.name(name));
        self.raw.contexts.push((id, crate::decl::ContextDecl { ty, volatile: false }));
    }

    fn input(&mut self, name: &str, ty: TyId<Poly>) {
        let name = self.name(name);
        self.raw.inputs.push((name, ty));
    }

    fn script(self, source: &str) -> (Checked, Script) {
        let script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_script(&decls, &mut types, &script);
        (Checked { typeck, types }, script)
    }

    /// A script parsed from `source`, then rewritten by `edit` before it is
    /// checked: a form the parser does not write yet is built so.
    fn script_edited(self, source: &str, edit: impl FnOnce(&mut Script)) -> (Checked, Script) {
        let mut script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        edit(&mut script);
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_script(&decls, &mut types, &script);
        (Checked { typeck, types }, script)
    }

    fn template(self, source: &str) -> (Checked, Template) {
        let template = acvus_ast::parse_template(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_template(&decls, &mut types, &template);
        (Checked { typeck, types }, template)
    }
}

/// The type of a script's tail.
fn tail(checked: &Checked, script: &Script) -> Ty {
    let tail = script.tail.as_ref().expect("the test's script has a tail");
    checked.typeck.types[&tail.id()]
}

/// The type of the `let` binding `name`.
fn binding(checked: &Checked, script: &Script, names: &Interner, name: &str) -> Ty {
    let name = names.intern(name);
    script
        .stmts
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::LetBind { binder, .. } | Stmt::LetUninit { binder, .. } if binder.name == name => {
                Some(checked.typeck.types[&binder.id])
            }
            _ => None,
        })
        .expect("the test's script binds the name")
}

fn is_int(checked: &Checked, ty: Ty, width: IntTy) -> bool {
    checked.kind(ty) == &TyKind::Int(width)
}

// Literals (types §5, operators §1)

/// types §5: an unsuffixed integer literal is `i64` where no use decides.
#[test]
fn an_unsuffixed_literal_no_use_decides_is_i64() {
    let (checked, script) = Fixture::new().script("1");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
}

/// A suffixed literal names its own width.
#[test]
fn a_suffixed_literal_has_its_width() {
    let (checked, script) = Fixture::new().script("1u8");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
}

/// types §5: a type variable nothing constrains is `!`; `[]` is `[!; 0]`.
#[test]
fn an_empty_list_is_an_array_of_never() {
    let (checked, script) = Fixture::new().script("[]");
    checked.admitted();
    let TyKind::Array(element, LenTerm::Known(0)) = checked.kind(tail(&checked, &script)) else {
        panic!("not an empty array");
    };
    assert_eq!(checked.kind(*element), &TyKind::Never);
}

// Names (types §6, programs §4)

/// A binding's term is what flows into it; a use of it names the binder.
#[test]
fn a_binding_takes_the_value_it_is_let_to() {
    let mut fx = Fixture::new();
    let i32_ty = fx.int(IntTy::I32);
    fx.context("n", i32_ty);
    let names = fx.names.clone();
    let (checked, script) = fx.script("let x = @n; x");
    checked.admitted();
    assert!(is_int(&checked, binding(&checked, &script, &names, "x"), IntTy::I32));
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I32));
}

/// A name that is no binding, declaration or input is refused.
#[test]
fn an_unknown_name_is_refused() {
    let fx = Fixture::new();
    let y = fx.name("y");
    let (checked, _) = fx.script("y");
    assert_eq!(checked.refusals(), vec![&RefusalKind::UnknownName { name: y }]);
}

/// programs §4: an input is a value the host passes, at its declared type.
#[test]
fn an_input_has_its_declared_type() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("flag", bool_ty);
    let (checked, script) = fx.script("$flag");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// programs §4: the program never assigns an input.
#[test]
fn an_input_is_never_assigned() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("flag", bool_ty);
    let flag = fx.name("flag");
    let (checked, _) = fx.script("$flag = true; 1");
    assert_eq!(checked.refusals(), vec![&RefusalKind::InputAssigned { input: flag }]);
}

/// programs §2: a script has no `fn`.
#[test]
fn a_script_has_no_fn() {
    let (checked, _) = Fixture::new().script("fn f() { 1 } 2");
    assert_eq!(checked.refusals(), vec![&RefusalKind::FnDeclInScript]);
}

// Operators (operators §2)

/// operators §2: `+` is a call of `core::add`; the operands' types decide
/// the impl, and the impl chooses the result.
#[test]
fn addition_reaches_the_impl_its_operands_decide() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::U8), TyKind::Int(IntTy::I64)]);
    let (checked, script) = fx.script("1 + 2");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
    let tail_site = script.tail.as_ref().unwrap().id();
    let chosen = &checked.typeck.reached[&tail_site];
    assert_eq!((chosen.declaration, chosen.r#impl), (add, Some(Impl::Concrete(1))));
}

/// operators §2 with types §5: a literal's width is decided by its use; an
/// operand of `u8` leaves one impl, and the literal takes `u8` from it.
#[test]
fn a_literal_operand_takes_the_width_its_use_decides() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::U8), TyKind::Int(IntTy::I64)]);
    let u8_ty = fx.int(IntTy::U8);
    fx.context("b", u8_ty);
    let (checked, script) = fx.script("@b + 1");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
}

/// operators §2: where no impl answers, the program is refused.
#[test]
fn an_operator_no_impl_answers_is_refused() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::I64)]);
    let (checked, _) = fx.script("true + false");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: add.name }]);
}

/// operators §2: `&&` is the language's own, on `bool`.
#[test]
fn logical_and_is_the_languages_own_on_bool() {
    let (checked, script) = Fixture::new().script("true && false");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    assert!(checked.typeck.reached.is_empty());
    let (checked, _) = Fixture::new().script("true && 1");
    assert!(matches!(checked.refusals()[..], [RefusalKind::Mismatch(_)]));
}

// Objects and fields (types §1-2, ownership §4)

/// types §1: an object the program writes is transparent, and a field
/// read of it is its field's type.
#[test]
fn a_written_objects_field_is_read() {
    let (checked, script) = Fixture::new().script("let o = { a: true, }; o.a");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// ownership §4, types §2: a store into a field of a binding declared
/// without a value joins that field into its object type.
#[test]
fn a_store_into_a_field_grows_the_bindings_object() {
    let fx = Fixture::new();
    let (x, y) = (fx.name("x"), fx.name("y"));
    let names = fx.names.clone();
    let (checked, script) = fx.script("let a; a.x = 1; a.y = true; a");
    checked.admitted();
    let TyKind::Object(ObjectTy::Structural(fields, ())) = checked.kind(binding(&checked, &script, &names, "a")) else {
        panic!("not a structural object");
    };
    let names: Vec<Astr> = fields.iter().map(|(name, _)| *name).collect();
    let mut expected = vec![x, y];
    expected.sort();
    assert_eq!(names, expected);
}

/// types §1, context §2: a context's type is opaque, so a field it lacks is
/// not grown into it.
#[test]
fn a_field_an_opaque_object_lacks_is_refused() {
    let mut fx = Fixture::new();
    let age = fx.name("age");
    let i64_ty = fx.int(IntTy::I64);
    let user = fx.ty(TyKind::Object(ObjectTy::Structural(Box::new([(age, i64_ty)]), ())));
    fx.context("user", user);
    let (checked, _) = fx.script("@user.name");
    assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque]);
}

// Control flow (control-flow, types §2-4)

/// types §3: the branches of an `if` enter one slot.
#[test]
fn branches_enter_one_slot() {
    let (checked, script) = Fixture::new().script("if true { 1u8 } else { 2 }");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
    let (checked, _) = Fixture::new().script("if true { 1 } else { false }");
    assert!(matches!(checked.refusals()[..], [RefusalKind::Mismatch(_)]));
}

/// control-flow, types §4: `return` is `!`, and its value enters the body's
/// result.
#[test]
fn return_is_never_and_enters_the_result() {
    let (checked, script) = Fixture::new().script("if true { return 1u8; }; 2");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
}

/// control-flow: a range iterates its integers by value.
#[test]
fn a_range_iterates_by_value() {
    let fx = Fixture::new();
    let (checked, script) = fx.script("for i in 0..3 { i; } 1");
    checked.admitted();
    let Stmt::For { id, binder, .. } = &script.stmts[0] else {
        panic!("not a for");
    };
    assert_eq!(checked.typeck.iteration[id], Iteration::Range);
    assert!(is_int(&checked, checked.typeck.types[&binder.id], IntTy::I64));
}

/// control-flow: a reference to an array iterates by reference; an array
/// by value.
#[test]
fn an_array_iterates_by_reference_through_a_reference_and_by_value_itself() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let array = fx.ty(TyKind::Array(i64_ty, LenTerm::Known(3)));
    fx.context("xs", array);
    let (checked, script) = fx.script("for x in &@xs { x; } for y in @xs { y; } 1");
    checked.admitted();
    let Stmt::For { id, binder, .. } = &script.stmts[0] else {
        panic!("not a for");
    };
    assert_eq!(checked.typeck.iteration[id], Iteration::ByRef(Mutability::Shared));
    let TyKind::Ref(Mutability::Shared, element) = checked.kind(checked.typeck.types[&binder.id]) else {
        panic!("not a shared reference");
    };
    assert!(is_int(&checked, element.ty(), IntTy::I64));
    let Stmt::For { id, binder, .. } = &script.stmts[1] else {
        panic!("not a for");
    };
    assert_eq!(checked.typeck.iteration[id], Iteration::ByValue);
    assert!(is_int(&checked, checked.typeck.types[&binder.id], IntTy::I64));
}

/// control-flow: nothing else is iterated.
#[test]
fn an_integer_is_not_iterated() {
    let (checked, _) = Fixture::new().script("for x in 1 { x; } 1");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NotIterable]);
}

/// control-flow: `?` on an `Option` is its payload, and leaves through
/// `Option`.
#[test]
fn try_on_an_option_is_its_payload() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let option = fx.ty(TyKind::Option(i64_ty));
    fx.context("o", option);
    let (checked, script) = fx.script("let x = @o?; Some(x)");
    checked.admitted();
    let TyKind::Option(payload) = checked.kind(tail(&checked, &script)) else {
        panic!("not an option");
    };
    assert!(is_int(&checked, *payload, IntTy::I64));
    assert!(matches!(checked.typeck.tries.values().next(), Some(TryExit::Option(_))));
}

/// control-flow: a `?` whose operand is neither a `Result` nor an `Option`
/// is refused.
#[test]
fn try_on_a_bool_is_refused() {
    let (checked, _) = Fixture::new().script("let x = true?; 1");
    assert!(checked.refusals().contains(&&RefusalKind::NotTryable));
}

// Patterns (patterns §1-4, types §2)

/// patterns §2: a pattern naming a variant grows a transparent enum.
#[test]
fn a_pattern_grows_a_transparent_enum() {
    let fx = Fixture::new();
    let (red, blue) = (fx.name("Red"), fx.name("Blue"));
    let names = fx.names.clone();
    let (checked, script) = fx.script("let c = Color::Red; match c { Color::Red => 1, Color::Blue => 2, }");
    checked.admitted();
    let TyKind::Enum { variants, .. } = checked.kind(binding(&checked, &script, &names, "c")) else {
        panic!("not an enum");
    };
    let tags: Vec<Astr> = variants.iter().map(|(tag, _)| *tag).collect();
    let mut expected = vec![red, blue];
    expected.sort();
    assert_eq!(tags, expected);
}

/// patterns §2: an enum not visible there does not grow; a pattern naming
/// a variant it lacks is refused.
#[test]
fn a_pattern_naming_a_variant_an_opaque_enum_lacks_is_refused() {
    let mut fx = Fixture::new();
    let color = QualifiedRef::root(fx.name("Color"));
    let red = fx.name("Red");
    let declared = fx.ty(TyKind::Enum {
        name: color,
        variants: Box::new([(red, None)]),
        label: (),
    });
    fx.context("c", declared);
    let (checked, _) = fx.script("match @c { Color::Red => 1, Color::Blue => 2, }");
    assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque]);
}

/// patterns §1: an object pattern binds its fields.
#[test]
fn an_object_pattern_binds_fields() {
    let mut fx = Fixture::new();
    let name = fx.name("name");
    let bool_ty = fx.ty(TyKind::Bool);
    let data = fx.ty(TyKind::Object(ObjectTy::Structural(Box::new([(name, bool_ty)]), ())));
    fx.context("data", data);
    let (checked, script) = fx.script("if let { name, } = @data { name } else { false }");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// types §2: `Some(x)` is `core`'s `Option`.
#[test]
fn some_is_an_option() {
    let (checked, script) = Fixture::new().script("Some(true)");
    checked.admitted();
    let TyKind::Option(payload) = checked.kind(tail(&checked, &script)) else {
        panic!("not an option");
    };
    assert_eq!(checked.kind(*payload), &TyKind::Bool);
}

// Calls (types §6-7, extern-fn §5)

/// types §6: a call of a name reaches the declaration its arguments'
/// types decide.
#[test]
fn a_call_reaches_the_overload_its_argument_decides() {
    let mut fx = Fixture::new();
    let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    let of_int = fx.pure(&[i64_ty], bool_ty);
    let of_bool = fx.pure(&[bool_ty], i64_ty);
    let first = QualifiedRef::qualified(fx.name("a"), fx.name("f"));
    let second = QualifiedRef::qualified(fx.name("b"), fx.name("f"));
    for (id, ty) in [(first, of_int), (second, of_bool)] {
        let scheme = Scheme {
            ty,
            param_names: fx.param_names(1),
            bounds: Box::new([]),
            impls: None,
            requires: Box::new([]),
        };
        fx.raw.functions.push((id, scheme));
    }
    let (checked, script) = fx.script("f(true)");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(checked.typeck.reached[&site].declaration, second);
}

/// types §6: several declarations left is a refusal.
#[test]
fn a_call_two_declarations_answer_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    let f = fx.name("f");
    for namespace in ["a", "b"] {
        let id = QualifiedRef::qualified(fx.name(namespace), f);
        let scheme = Scheme {
            ty,
            param_names: fx.param_names(1),
            bounds: Box::new([]),
            impls: None,
            requires: Box::new([]),
        };
        fx.raw.functions.push((id, scheme));
    }
    let (checked, _) = fx.script("f(1)");
    assert!(matches!(checked.refusals()[..], [RefusalKind::SeveralDeclarations { .. }]));
}

/// `left | f` is `f(left)`.
#[test]
fn a_pipe_calls_its_right_with_its_left() {
    let mut fx = Fixture::new();
    let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    let ty = fx.pure(&[i64_ty], bool_ty);
    fx.declare("positive", ty, &[]);
    let (checked, script) = fx.script("1 | positive");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// operators §2: `.unwrap()` is `core::unwrap`, `unwrap<W, T>(W) -> T` at
/// `Option<T>`.
#[test]
fn unwrap_reaches_the_lang_item() {
    let mut fx = Fixture::new();
    let unwrap = fx.raw.core.unwrap;
    let t = fx.var(0);
    let option_t = fx.ty(TyKind::Option(t));
    let at_option = fx.function(&[option_t], t, EffectTerm::Known(crate::ty::Effect::pure()));
    fx.lang_item(unwrap, &[at_option]);
    let bool_ty = fx.ty(TyKind::Bool);
    let option = fx.ty(TyKind::Option(bool_ty));
    fx.context("o", option);
    let (checked, script) = fx.script("@o.unwrap()");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

// Lambdas (functions §2-4)

/// functions §2, D27: a lambda's parameters start fresh and are decided by
/// the slot it enters and its uses; a name from outside is a capture by
/// reference.
#[test]
fn a_lambda_captures_by_reference_and_takes_its_arguments_types() {
    let fx = Fixture::new();
    let y = fx.name("y");
    let (checked, script) = fx.script("let y = true; let f = |x| -> y; f(1u8)");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    assert_eq!(
        &closure.captures[..],
        &[Captured {
            name: y,
            capture: Capture::Ref(Mutability::Shared)
        }]
    );
    assert_eq!(closure.calls, Calls::Many);
}

/// functions §4: a lambda may not recurse.
#[test]
fn a_lambda_may_not_recurse() {
    let (checked, _) = Fixture::new().script("let f; f = |x| -> f(x); 1");
    assert!(checked.refusals().contains(&&RefusalKind::LambdaRecursion));
}

// Places (operators §3, §6)

/// operators §6: `*r = v` needs a `&mut`.
#[test]
fn a_store_through_a_shared_reference_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let (checked, _) = fx.script("let r = &@n; *r = 1; 1");
    assert!(checked.refusals().contains(&&RefusalKind::StoreThroughShared));
}

/// operators §6: `*r` of a `&T` is a `T`.
#[test]
fn a_deref_reads_the_referent() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let (checked, script) = fx.script("let r = &@n; *r");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
}

/// operators §3: `a[i]` of an array is its element.
#[test]
fn an_arrays_element_is_indexed() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    let array = fx.ty(TyKind::Array(bool_ty, LenTerm::Known(2)));
    fx.context("xs", array);
    let (checked, script) = fx.script("@xs[0]");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// operators §3: `a[i]` of another container reaches its `as_slice`.
#[test]
fn a_containers_element_is_indexed_through_as_slice() {
    let mut fx = Fixture::new();
    let vec_id = fx.vec_with_as_slice();
    let as_slice = fx.raw.core.as_slice;
    let bool_ty = fx.ty(TyKind::Bool);
    let vec_bool = fx.vec_of(vec_id, bool_ty);
    fx.context("v", vec_bool);
    let (checked, script) = fx.script("@v[0]");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    let Expr::Index { callee_id, .. } = script.tail.as_deref().unwrap() else {
        panic!("not an index");
    };
    assert_eq!(checked.typeck.reached[callee_id].declaration, as_slice);
}

// Contexts (context §2-3, ownership §5)

/// ownership §5: a run's result has a type with no positions.
#[test]
fn a_runs_result_holds_no_loan() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let (checked, _) = fx.script("&@n");
    assert_eq!(checked.refusals(), vec![&RefusalKind::HoldsLoans]);
}

/// context §3: a value stored into a context outlives the run, so its type
/// has no positions.
#[test]
fn a_context_of_a_function_type_is_never_stored() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let function = fx.pure(&[i64_ty], i64_ty);
    fx.context("f", function);
    let (checked, _) = fx.script("@f = @f; 1");
    assert_eq!(checked.refusals(), vec![&RefusalKind::HoldsLoans]);
}

/// context: a context not declared is refused.
#[test]
fn an_undeclared_context_is_refused() {
    let fx = Fixture::new();
    let x = QualifiedRef::root(fx.name("x"));
    let (checked, _) = fx.script("@x");
    assert_eq!(checked.refusals(), vec![&RefusalKind::UnknownContext { context: x }]);
}

// Templates (programs §1, operators §2)

/// programs §1: `{{ v }}` formats `v` through `core::display`.
#[test]
fn an_interpolation_reaches_display() {
    let mut fx = Fixture::new();
    fx.display(&[TyKind::Int(IntTy::I64)]);
    let display = fx.raw.core.display;
    let (checked, template) = fx.template("{{ 42 }}");
    checked.admitted();
    let Stmt::Append { id, .. } = &template.body[0] else {
        panic!("not an append");
    };
    assert_eq!(checked.typeck.reached[id].declaration, display);
}

/// operators §2: where no `display` impl answers, the template is refused.
#[test]
fn an_interpolation_no_display_answers_is_refused() {
    let mut fx = Fixture::new();
    fx.display(&[TyKind::Int(IntTy::I64)]);
    let (checked, _) = fx.template("{{ true }}");
    assert!(matches!(checked.refusals()[..], [RefusalKind::NoDeclaration { .. }]));
}

/// functions §2: `move` captures each name by move.
#[test]
fn a_move_lambda_captures_by_move() {
    let fx = Fixture::new();
    let y = fx.name("y");
    let (checked, _) = fx.script("let y = true; let f = move |x| -> y; f(1u8)");
    checked.admitted();
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    assert_eq!(
        &closure.captures[..],
        &[Captured {
            name: y,
            capture: Capture::Move
        }]
    );
    assert_eq!(closure.calls, Calls::Many, "a `bool` copies, so nothing is moved out");
}

/// functions §3: a `move` closure whose body moves a captured value out is
/// called once.
#[test]
fn a_move_lambda_that_moves_a_capture_out_is_once() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    fx.context("s", string);
    let (checked, script) = fx.script("let s = @s; let f = move |x| -> s; 1");
    checked.admitted();
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    assert_eq!(closure.calls, Calls::Once);
    let Stmt::LetBind { expr, .. } = &script.stmts[1] else {
        panic!("not a let");
    };
    let TyKind::Fn { calls, .. } = checked.kind(checked.typeck.types[&expr.id()]) else {
        panic!("not a function");
    };
    assert_eq!(*calls, Calls::Once);
}

/// ownership §5: a `move` closure holds the strongest loan its captured
/// values hold; a lambda capturing by reference holds that reference.
#[test]
fn a_closure_holds_the_loans_of_its_captures() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let (checked, script) = fx.script("let r = &mut @n; let f = move |x| -> r; let g = |x| -> r; 1");
    checked.admitted();
    let holds = |at: usize| {
        let Stmt::LetBind { expr, .. } = &script.stmts[at] else {
            panic!("not a let");
        };
        let TyKind::Fn { holds, .. } = checked.kind(checked.typeck.types[&expr.id()]) else {
            panic!("not a function");
        };
        *holds
    };
    assert_eq!(holds(1), Holds::Mut);
    assert_eq!(holds(2), Holds::Shared);
}

/// control-flow: `break` and `continue` leave a loop; one outside every
/// loop, or inside a lambda inside one, is refused.
#[test]
fn break_outside_a_loop_is_refused() {
    let (checked, _) = Fixture::new().script("while true { break; } 1");
    checked.admitted();
    let (checked, _) = Fixture::new().script("break; 1");
    assert_eq!(checked.refusals(), vec![&RefusalKind::OutsideLoop]);
    let (checked, _) = Fixture::new().script("while true { let f = |x| -> { continue; x }; } 1");
    assert_eq!(checked.refusals(), vec![&RefusalKind::OutsideLoop]);
}

// Casts (operators §5)

/// operators §5: `as` casts between the numbers, `bool` and `char` as Rust
/// does; its value has the type it names.
#[test]
fn as_casts_as_rust_casts() {
    let (checked, script) = Fixture::new().script("1u8 as i64");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(
        checked.typeck.casts[&site],
        super::AsCast {
            from: crate::ty::CastTy::Int(IntTy::U8),
            to: crate::ty::CastTy::Int(IntTy::I64)
        }
    );
}

/// operators §5: nothing is cast to `bool`, and `as` names only the
/// numbers, `bool` and `char`.
#[test]
fn an_as_rust_does_not_admit_is_refused() {
    let (checked, _) = Fixture::new().script("1u8 as bool");
    assert_eq!(
        checked.refusals(),
        vec![&RefusalKind::CastNotAdmitted {
            from: Some(crate::ty::CastTy::Int(IntTy::U8)),
            to: crate::ty::CastTy::Bool
        }]
    );
    let fx = Fixture::new();
    let string = fx.name("String");
    let (checked, _) = fx.script("1 as String");
    assert_eq!(checked.refusals(), vec![&RefusalKind::UnknownCastTarget { name: string }]);
}

// Coercions (types §3, operators §5)

/// operators §5: a reference to a container reads as its view where the
/// slot asks for it; the coercion is recorded at the argument, with its
/// step.
#[test]
fn a_view_coercion_is_recorded_at_the_argument() {
    let mut fx = Fixture::new();
    let vec_id = fx.vec_with_as_slice();
    let as_slice = fx.raw.core.as_slice;
    let bool_ty = fx.ty(TyKind::Bool);
    let slice = fx.ty(TyKind::Slice(bool_ty));
    let shared_slice = fx.reference(Mutability::Shared, slice);
    let unit = fx.ty(TyKind::Unit);
    let takes_slice = fx.pure(&[shared_slice], unit);
    fx.declare("first", takes_slice, &[]);
    let vec_bool = fx.vec_of(vec_id, bool_ty);
    fx.context("v", vec_bool);
    let (checked, script) = fx.script("first(&@v)");
    checked.admitted();
    let Expr::FuncCall { args, .. } = script.tail.as_deref().unwrap() else {
        panic!("not a call");
    };
    let steps = &checked.typeck.conversions[&args[0].id()];
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].via.declaration, as_slice);
    let TyKind::Ref(Mutability::Shared, viewed) = checked.kind(steps[0].to) else {
        panic!("a view is a reference");
    };
    assert!(matches!(checked.kind(viewed.ty()), TyKind::Slice(_)));
}

// Comparisons (operators §4)

/// operators §4: a structural type compares part by part, each part
/// through its `core::eq` impl.
#[test]
fn a_tuple_compares_part_by_part() {
    let mut fx = Fixture::new();
    let eq = fx.raw.core.eq;
    let bool_ty = fx.ty(TyKind::Bool);
    let impls: Vec<TyId<Poly>> = [TyKind::Int(IntTy::I64), TyKind::Bool]
        .into_iter()
        .map(|kind| {
            let t = fx.ty(kind);
            let shared = fx.reference(Mutability::Shared, t);
            fx.pure(&[shared, shared], bool_ty)
        })
        .collect();
    fx.lang_item(eq, &impls);
    let (checked, script) = fx.script("(1, true) == (2, false)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    let super::Comparison::Parts(parts) = &checked.typeck.comparisons[&site] else {
        panic!("a tuple compares its parts");
    };
    let reached: Vec<Option<Impl>> = parts
        .iter()
        .map(|part| match part {
            super::Comparison::Reached(chosen) => chosen.r#impl,
            super::Comparison::Parts(_) => panic!("an integer and a bool reach an impl"),
        })
        .collect();
    assert_eq!(reached, vec![Some(Impl::Concrete(0)), Some(Impl::Concrete(1))]);
    assert!(!checked.typeck.reached.contains_key(&site));
}

// Forms decided by their operand's head (types §6)

/// types §6: a field read through a reference is decided once the object's
/// type is known, which here is only after the lambda is called.
#[test]
fn a_field_through_a_reference_waits_for_its_objects_type() {
    let mut fx = Fixture::new();
    let a = fx.name("a");
    let bool_ty = fx.ty(TyKind::Bool);
    let object = fx.ty(TyKind::Object(ObjectTy::Structural(Box::new([(a, bool_ty)]), ())));
    fx.context("o", object);
    let (checked, script) = fx.script("let f = |r| -> r.a; f(&@o)");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// patterns §1: through a reference, Rust's default binding modes lend
/// what a pattern binds.
#[test]
fn a_pattern_through_a_reference_binds_references() {
    let fx = Fixture::new();
    let names = fx.names.clone();
    let (checked, script) = fx.script("let f = |p| -> match p { (x, y) => x, }; let r = f(&(1u8, true)); 1");
    checked.admitted();
    let TyKind::Ref(Mutability::Shared, target) = checked.kind(binding(&checked, &script, &names, "r")) else {
        panic!("a binding through a `&` is a `&`");
    };
    assert!(is_int(&checked, target.ty(), IntTy::U8));
}

/// operators §3: an index is decided by its container's type, known here
/// only once the lambda is called.
#[test]
fn an_index_waits_for_its_containers_type() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    let array = fx.ty(TyKind::Array(bool_ty, LenTerm::Known(2)));
    fx.context("xs", array);
    let (checked, script) = fx.script("let g = |xs| -> xs[0]; g(@xs)");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
}

/// operators §6: a store through a reference known only later to be
/// shared is refused once it is known.
#[test]
fn a_store_through_a_reference_known_later_to_be_shared_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let (checked, _) = fx.script("let h = |r| -> { *r = 1; 2 }; h(&@n)");
    assert!(checked.refusals().contains(&&RefusalKind::StoreThroughShared));
}

/// types §6: a decision reads a term only once no pending decision may
/// change it, and a term linked to a pending slot is that slot. `==` makes
/// `y` one with the `if`'s slot, which `g(w)` raises from `u8` to `i64`
/// once `w` is known, so `f(y)` reaches `f` at `i64`.
#[test]
fn a_term_linked_to_a_pending_slot_waits_for_it() {
    let mut fx = Fixture::new();
    let (u8_ty, i64_ty, bool_ty) = (fx.int(IntTy::U8), fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    let widening = fx.pure(&[u8_ty], i64_ty);
    let widen = fx.declare("widen", widening, &[]);
    fx.raw.casts.push(CastRule {
        from: u8_ty,
        to: i64_ty,
        via: widen,
    });
    let of_bool = fx.pure(&[bool_ty], i64_ty);
    fx.overload("a", "g", widening);
    fx.overload("b", "g", of_bool);
    let (takes_u8, takes_i64) = (fx.pure(&[u8_ty], bool_ty), fx.pure(&[i64_ty], bool_ty));
    fx.overload("a", "f", takes_u8);
    let wide = fx.overload("b", "f", takes_i64);
    let eq = fx.raw.core.eq;
    let equalities: Vec<TyId<Poly>> = [u8_ty, i64_ty]
        .into_iter()
        .map(|compared| {
            let shared = fx.reference(Mutability::Shared, compared);
            fx.pure(&[shared, shared], bool_ty)
        })
        .collect();
    fx.lang_item(eq, &equalities);
    let (checked, script) = fx.script("let y; let w; let z = y == (if true { 1u8 } else { g(w) }); w = 2u8; f(y)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(checked.typeck.reached[&site].declaration, wide);
}

/// `left + right` of a script's tail rewritten to `left ^ right`, which the
/// parser does not write yet.
fn xor_tail(script: &mut Script) {
    let Some(Expr::BinaryOp { op, .. }) = script.tail.as_deref_mut() else {
        panic!("the test's tail is a binary operator");
    };
    *op = BinOp::Xor;
}

/// operators §2: `^` is the language's own on `bool` and the integers,
/// and on no other type.
#[test]
fn xor_is_the_languages_own_on_bool_and_the_integers() {
    let (checked, script) = Fixture::new().script_edited("true + false", xor_tail);
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    assert!(checked.typeck.reached.is_empty());
    let (checked, script) = Fixture::new().script_edited("1u8 + 2", xor_tail);
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
    let (checked, _) = Fixture::new().script_edited("(1, 2) + (3, 4)", xor_tail);
    assert_eq!(checked.refusals(), vec![&RefusalKind::NotBoolOrInteger]);
}

/// types §3, §7: a `return` and the tail each enter the run's result at
/// their own site. The result is the two's join up the cast tree, and the
/// `u8` returned is converted where it is returned.
#[test]
fn a_return_and_the_tail_each_enter_the_runs_result() {
    let mut fx = Fixture::new();
    let (u8_ty, i64_ty) = (fx.int(IntTy::U8), fx.int(IntTy::I64));
    let widening = fx.pure(&[u8_ty], i64_ty);
    let widen = fx.declare("widen", widening, &[]);
    fx.raw.casts.push(CastRule {
        from: u8_ty,
        to: i64_ty,
        via: widen,
    });
    let (checked, script) = fx.script("if true { return 1u8; }; 2i64");
    checked.admitted();
    let Stmt::Expr(Expr::If { then_body, .. }) = &script.stmts[0] else {
        panic!("not an if");
    };
    let Stmt::Expr(Expr::Return { value, .. }) = &then_body[0] else {
        panic!("not a return");
    };
    let steps = &checked.typeck.conversions[&value.id()];
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].via.declaration, widen);
    assert!(is_int(&checked, steps[0].to, IntTy::I64));
    let tail_site = script.tail.as_ref().unwrap().id();
    assert!(!checked.typeck.conversions.contains_key(&tail_site));
}

/// operators §5: a view step names the impl of the view it is; here the
/// second `as_slice` impl.
#[test]
fn a_view_step_names_its_impl() {
    let mut fx = Fixture::new();
    let other_id = QualifiedRef::root(fx.name("Other"));
    fx.raw.user_types.push(UserTypeDecl {
        id: other_id,
        type_params: Box::new([]),
        specializes: Box::new([]),
        effect_params: 0,
        identity_params: 0,
        lifetimes: 0,
    });
    let other = fx.ty(TyKind::UserDefined {
        id: other_id,
        type_args: Box::new([]),
        effect_args: Box::new([]),
        identity_args: Box::new([]),
    });
    let bool_ty = fx.ty(TyKind::Bool);
    let shared_other = fx.reference(Mutability::Shared, other);
    let bools = fx.ty(TyKind::Slice(bool_ty));
    let shared_bools = fx.reference(Mutability::Shared, bools);
    let at_other = fx.pure(&[shared_other], shared_bools);
    let (vec_id, at_vec) = fx.vec_and_its_view();
    let as_slice = fx.raw.core.as_slice;
    fx.lang_item(as_slice, &[at_other, at_vec]);
    let unit = fx.ty(TyKind::Unit);
    let takes_slice = fx.pure(&[shared_bools], unit);
    fx.declare("first", takes_slice, &[]);
    let vec_bool = fx.vec_of(vec_id, bool_ty);
    fx.context("v", vec_bool);
    let (checked, script) = fx.script("first(&@v)");
    checked.admitted();
    let Expr::FuncCall { args, .. } = script.tail.as_deref().unwrap() else {
        panic!("not a call");
    };
    let steps = &checked.typeck.conversions[&args[0].id()];
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].via.declaration, as_slice);
    assert_eq!(steps[0].via.r#impl, Some(Impl::Concrete(1)));
}

/// control-flow, types §6: a block whose last statement's type is `!` is
/// `!`, where that type becomes known only once its call is reached.
#[test]
fn a_block_ending_in_a_call_of_type_never_is_never() {
    let mut fx = Fixture::new();
    let never = fx.ty(TyKind::Never);
    let aborts = fx.pure(&[], never);
    fx.declare("abort", aborts, &[]);
    let (checked, script) = fx.script("if true { 1u8 } else { abort(); }");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
    let Some(Expr::If { else_branch, .. }) = script.tail.as_deref() else {
        panic!("not an if");
    };
    assert!(matches!(else_branch.as_deref(), Some(ElseBranch::Else { tail: None, .. })));
}
