//! Each test states the rule its expected result comes from. Programs are
//! parsed with `acvus-ast`; declarations go through `verify`.

use acvus_ast::{AstId, ElseBranch, Expr, Script, Stmt, Template};
use acvus_utils::{Astr, Interner, Namespace, QualifiedRef};

use super::{
    Adjustment, Capture, CaptureOf, Captured, Exceeding, FrozenMismatch, Impl, Inputs, Interfaces, Iteration, Name, RefusalKind, TryExit, Typeck, UnitCall,
    check_script, check_template,
};
use super::decision::variables_of;
use crate::decl::{
    CastRule, Core, Declarations, DeclaredFacts, GenericFunction, ImplKind, ImplOf, LoopMarkers, RawDeclarations, Scheme, TyVarBound, UserTypeDecl,
};
use crate::infer::{Mismatch, ShapeCoercion};
use crate::ty::structural::PartKey;
use crate::ty::{IdentityTerm, Mark, MarkTerm};
use crate::ty::{
    Calls, Concrete, EffectTerm, FlowTerm, Flows, Holds, IntTy, LenTerm, Mutability, Poly, Ty, TyId, TyKind, Types,
};

/// The impl at `at` of `generic`, a declaration of its own named `generic#at`.
fn impl_id(names: &Interner, generic: QualifiedRef, at: usize) -> QualifiedRef {
    QualifiedRef::qualified(generic.namespace, names.intern(&format!("{}#{at}", names.resolve(generic.name))))
}

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
    /// Each input `$x` with its type, at which a run's inputs are fixed.
    inputs: Vec<(Astr, TyId<Poly>)>,
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

    /// The steps of the concrete coercion typeck recorded at `site`.
    fn steps(&self, site: AstId) -> &[super::Conversion<Ty>] {
        match &self.typeck.coercions[&site] {
            super::Coercion::Steps(steps) => steps,
            other => panic!("a concrete value is coerced by steps, not {other:?}"),
        }
    }

    fn admitted(&self) {
        assert!(self.typeck.refusals.is_empty(), "refused: {:?}", self.typeck.refusals);
    }
}

impl Fixture {
    fn new() -> Self {
        Self::with_names(Interner::new())
    }

    /// A fixture interning into `names`, which another fixture may share,
    /// so that the names of the two are one.
    fn with_names(names: Interner) -> Self {
        let core = Core::of(&names);
        Self {
            names,
            raw: RawDeclarations {
                generic_functions: Vec::new(),
                types: Types::new(),
                functions: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                entries: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
            inputs: Vec::new(),
        }
    }

    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn namespace(&self, path: &[&str]) -> Namespace {
        let path: Vec<Astr> = path.iter().map(|segment| self.names.intern(segment)).collect();
        self.names.namespace(&path)
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
        self.ty(TyKind::Ref(mutability, target))
    }

    fn function(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, effect: EffectTerm<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect,
            flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
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
            impl_of: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        self.raw.functions.push((id, scheme));
        id
    }

    /// A plain extern fn `name` in `namespace`, one of the declarations a
    /// bare `name` may reach.
    fn overload(&mut self, namespace: &str, name: &str, ty: TyId<Poly>) -> QualifiedRef {
        let namespace = self.namespace(&[namespace]);
        self.declare_at(namespace, name, ty)
    }

    /// A plain extern fn `name` at `namespace`, a path of any depth.
    fn declare_at(&mut self, namespace: Namespace, name: &str, ty: TyId<Poly>) -> QualifiedRef {
        let id = QualifiedRef::qualified(namespace, self.name(name));
        let scheme = Scheme {
            ty,
            param_names: self.param_names(self.arity(ty)),
            bounds: Box::new([]),
            impl_of: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
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
            carries_effect: false,
            carries_identity: false,
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
            type_args: Box::new([element]),
            effect: None,
            identity: None,
            marked: MarkTerm::Known(Mark::Unmarked),
        })
    }

    /// A lang item's concrete impls; its generic function is the
    /// compiler's (operators §2).
    fn lang_item(&mut self, item: QualifiedRef, impls: &[TyId<Poly>]) {
        for (at, ty) in impls.iter().enumerate() {
            let scheme = Scheme {
                ty: *ty,
                bounds: vec![TyVarBound::Any; variables_of(&self.raw.types, &[*ty])].into(),
                param_names: self.param_names(self.arity(*ty)),
                impl_of: Some(ImplOf { generic_function: item, kind: ImplKind::Concrete }),
                requires: Box::new([]),
                facts: Default::default(),
                runs: Box::new([]),
                awaits: false,
                specialized: Box::new([]),
            };
            self.raw.functions.push((impl_id(&self.names, item, at), scheme));
        }
    }

    /// A lang item's concrete impls, each declaring `facts`.
    fn lang_item_stating(&mut self, item: QualifiedRef, impls: &[TyId<Poly>], facts: DeclaredFacts) {
        for (at, ty) in impls.iter().enumerate() {
            let scheme = Scheme {
                ty: *ty,
                bounds: vec![TyVarBound::Any; variables_of(&self.raw.types, &[*ty])].into(),
                param_names: self.param_names(self.arity(*ty)),
                impl_of: Some(ImplOf { generic_function: item, kind: ImplKind::Concrete }),
                requires: Box::new([]),
                facts: facts.clone(),
                runs: Box::new([]),
                awaits: false,
                specialized: Box::new([]),
            };
            self.raw.functions.push((impl_id(&self.names, item, at), scheme));
        }
    }

    /// Impls of `clone<T>(&T) -> T` at each of `types`.
    fn clones(&mut self, types: &[TyId<Poly>]) {
        let impls: Vec<TyId<Poly>> = types
            .iter()
            .map(|ty| {
                let shared = self.reference(Mutability::Shared, *ty);
                self.pure(&[shared], *ty)
            })
            .collect();
        let item = self.raw.core.clone;
        self.lang_item(item, &impls);
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
        self.inputs.push((name, ty));
    }

    fn script(self, source: &str) -> (Checked, Script) {
        self.script_calling(&Interfaces::new(), source)
    }

    /// `source` checked with its inputs fixed, its calls reaching the
    /// units of `interfaces` too.
    fn script_calling(self, interfaces: &Interfaces, source: &str) -> (Checked, Script) {
        let script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, interfaces, Inputs::Fixed(&inputs), &mut types, &script);
        (Checked { typeck, types }, script)
    }

    /// `source` checked with its inputs fixed, its calls reaching `units`,
    /// each a script at `namespace::name` checked first with its inputs open
    /// against the same declarations, its interface what the call reaches.
    fn script_with_units(self, units: &[(&[&str], &str, &str)], source: &str) -> (Checked, Script, Vec<QualifiedRef>) {
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut interfaces = Interfaces::new();
        let mut names = Vec::new();
        for (namespace, name, unit_source) in units {
            let unit_script = acvus_ast::parse_script(&self.names, unit_source).expect("the unit's source parses");
            let mut unit_types = Types::new();
            let checked = check_script(&self.names, &decls, &interfaces, Inputs::Open, &mut unit_types, &unit_script);
            assert!(checked.refusals.is_empty(), "the unit {name} is refused: {:?}", checked.refusals);
            let interface = checked.interface.expect("an open check gives its interface");
            let path: Vec<Astr> = namespace.iter().map(|segment| self.names.intern(segment)).collect();
            let unit = QualifiedRef::qualified(self.names.namespace(&path), self.names.intern(name));
            interfaces.set(unit, interface);
            names.push(unit);
        }
        let script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, &interfaces, Inputs::Fixed(&inputs), &mut types, &script);
        (Checked { typeck, types }, script, names)
    }

    /// Each of `sources` checked alone with its inputs open against one
    /// verification of the declarations, and its interface.
    fn interfaces(self, sources: &[&str]) -> Vec<super::Interface> {
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        sources
            .iter()
            .map(|source| {
                let script = acvus_ast::parse_script(&self.names, source).expect("the unit's source parses");
                let checked = check_script(&self.names, &decls, &Interfaces::new(), Inputs::Open, &mut Types::new(), &script);
                assert!(checked.refusals.is_empty(), "{source} is refused: {:?}", checked.refusals);
                checked.interface.expect("an open check gives its interface")
            })
            .collect()
    }

    fn template(self, source: &str) -> (Checked, Template) {
        let template = acvus_ast::parse_template(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_template(&self.names, &decls, &Interfaces::new(), Inputs::Fixed(&inputs), &mut types, &template);
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

/// The type recorded at an argument's site.
fn args_ty(checked: &Checked, arg: &Expr) -> Ty {
    checked.typeck.types[&arg.id()]
}

/// The binder of the `let` at statement `at`.
fn let_binder(script: &Script, at: usize) -> acvus_ast::AstId {
    match &script.stmts[at] {
        Stmt::LetBind { binder, .. } | Stmt::LetUninit { binder, .. } => binder.id,
        other => panic!("statement {at} is not a `let`: {other:?}"),
    }
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
    let TyKind::Array(element, LenTerm::Known(0), _) = checked.kind(tail(&checked, &script)) else {
        panic!("not an empty array");
    };
    assert_eq!(checked.kind(*element), &TyKind::Never);
}

/// types §5, §6: `r = &mut *r` gives `r`'s slot a value its own deref
/// makes, so the slot's meeting and the deref wait on each other, a cycle
/// through decisions. The deref's target is that decision's to bind, no
/// source of the cycle, so it takes no default before the cycle breaks on
/// `&mut x`, the value from outside it: `*r` is `i64`.
#[test]
fn a_reference_given_a_reborrow_of_itself_keeps_its_target() {
    let (checked, script) = Fixture::new().script("let x = 1; let r = &mut x; r = &mut *r; *r");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
}

/// The same where `r` is given itself.
#[test]
fn a_reference_given_itself_keeps_its_target() {
    let (checked, script) = Fixture::new().script("let x = 1; let r = &mut x; r = r; *r");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
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

/// programs §4: an input is a parameter of the run; fixed, `$x` enters
/// its given type, and `$x + 1` is an `i64` where `x` is.
#[test]
fn a_fixed_input_enters_its_given_type() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::I64)]);
    let i64_ty = fx.int(IntTy::I64);
    fx.input("x", i64_ty);
    let (checked, script) = fx.script("$x + 1");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
}

/// programs §4: with its inputs fixed, a `$x` not given is no input.
#[test]
fn an_input_not_given_is_refused() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::I64)]);
    let x = fx.name("x");
    let (checked, _) = fx.script("$x + 1");
    assert_eq!(checked.refusals(), vec![&RefusalKind::UnknownName { name: x }]);
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

/// The kind of the `let` binding `x`'s type: what a part read gave it.
fn read_of_x(source: &str, slice: Option<Mutability>) -> (Checked, Ty) {
    let mut fx = Fixture::new();
    if let Some(held) = slice {
        let int = fx.int(IntTy::I64);
        let element = fx.reference(Mutability::Mut, int);
        let slice_ty = fx.ty(TyKind::Slice(element));
        let lent = fx.reference(held, slice_ty);
        fx.input("s", lent);
    }
    let names = fx.names.clone();
    let (checked, script) = fx.script(source);
    checked.admitted();
    let x = binding(&checked, &script, &names, "x");
    (checked, x)
}

fn is_ref_to_i64(checked: &Checked, ty: Ty, mutability: Mutability) -> bool {
    matches!(checked.kind(ty), TyKind::Ref(made, target) if *made == mutability && is_int(checked, *target, IntTy::I64))
}

/// D108 rule 1: a `&mut` element read through a shared slice is a shared
/// reborrow.
#[test]
fn an_index_of_a_shared_slice_of_mut_references_reads_shared() {
    let (checked, x) = read_of_x("let x = $s[0]; *x", Some(Mutability::Shared));
    assert!(is_ref_to_i64(&checked, x, Mutability::Shared), "{:?}", checked.kind(x));
}

/// D108 rule 1: through a `&mut` slice the element keeps its own
/// mutability.
#[test]
fn an_index_of_a_mut_slice_of_mut_references_reads_mut() {
    let (checked, x) = read_of_x("let x = $s[0]; *x", Some(Mutability::Mut));
    assert!(is_ref_to_i64(&checked, x, Mutability::Mut), "{:?}", checked.kind(x));
}

/// D108 rule 1: the path's mutability is the least along it, so a `&`
/// anywhere before a `&mut` field makes its read shared.
#[test]
fn a_mut_field_read_past_a_shared_reference_reads_shared() {
    let source = "let v = 1; let o = { a: { b: &mut v, }, }; let r = &o; let x = r.a.b; *x";
    let (checked, x) = read_of_x(source, None);
    assert!(is_ref_to_i64(&checked, x, Mutability::Shared), "{:?}", checked.kind(x));
}

/// The type the name `b`, bound by the first arm of the script's tail
/// `match`, the first field of an object pattern, has.
fn first_field_binding(checked: &Checked, script: &Script) -> Ty {
    let Some(Expr::Match { arms, .. }) = script.tail.as_deref() else {
        panic!("the test's script ends in a match");
    };
    let acvus_ast::Pattern::Object { fields, .. } = &arms[0].pattern else {
        panic!("the first arm is an object pattern");
    };
    let acvus_ast::Pattern::Binding { id, .. } = &fields[0].pattern else {
        panic!("the first field binds a name");
    };
    checked.typeck.types[id]
}

/// `source`, admitted, and what `first_field_binding` finds.
fn bound_by_match(source: &str) -> (Checked, Ty) {
    let (checked, script) = Fixture::new().script(source);
    checked.admitted();
    let b = first_field_binding(&checked, &script);
    (checked, b)
}

/// D108 rule 1, patterns §1: a name binding a `&mut` part past a `&` is a
/// shared reborrow.
#[test]
fn a_mut_part_bound_past_a_shared_reference_binds_shared() {
    let (checked, b) = bound_by_match("let v = 1; let o = { b: &mut v, }; match &o { { b, } => *b, }");
    assert!(is_ref_to_i64(&checked, b, Mutability::Shared), "{:?}", checked.kind(b));
}

/// D108 rule 1, patterns §1: past a `&mut` the part keeps its own
/// mutability.
#[test]
fn a_mut_part_bound_past_a_mut_reference_binds_mut() {
    let (checked, b) = bound_by_match("let v = 1; let o = { b: &mut v, }; match &mut o { { b, } => *b, }");
    assert!(is_ref_to_i64(&checked, b, Mutability::Mut), "{:?}", checked.kind(b));
}

/// D108 rule 1, patterns §1: a `&` part bound past a `&mut` is the `&`
/// itself, the least along the path, not a `&mut` of it.
#[test]
fn a_shared_part_bound_past_a_mut_reference_binds_shared() {
    let (checked, b) = bound_by_match("let v = 1; let o = { b: &v, }; match &mut o { { b, } => *b, }");
    assert!(is_ref_to_i64(&checked, b, Mutability::Shared), "{:?}", checked.kind(b));
}

/// patterns §1: a part that is no reference, bound past a `&mut`, is lent
/// at the path's mutability.
#[test]
fn a_value_part_bound_past_a_mut_reference_is_lent_mut() {
    let (checked, b) = bound_by_match("let o = { b: 1, }; match &mut o { { b, } => *b, }");
    assert!(is_ref_to_i64(&checked, b, Mutability::Mut), "{:?}", checked.kind(b));
}

/// D108 rule 1: a store target is the part itself, not a read of it.
#[test]
fn a_store_into_a_mut_field_through_a_mut_reference_keeps_its_type() {
    let fx = Fixture::new();
    let (checked, script) = fx.script("let v = 1; let w = 2; let o = { f: &mut v, }; let r = &mut o; r.f = &mut w; *o.f");
    checked.admitted();
    let Some(Stmt::Store { place, .. }) = script.stmts.iter().find(|stmt| matches!(stmt, Stmt::Store { .. })) else {
        panic!("the script stores");
    };
    assert!(is_ref_to_i64(&checked, checked.typeck.types[&place.id()], Mutability::Mut));
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
    let TyKind::Object { fields, label: () } = checked.kind(binding(&checked, &script, &names, "a")) else {
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
    let user = fx.ty(TyKind::Object { fields: Box::new([(age, i64_ty)]), label: () });
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

/// control-flow, types §4: `return` gives no value, and its value enters
/// the body's result.
#[test]
fn return_gives_no_value_and_enters_the_result() {
    let (checked, script) = Fixture::new().script("if true { return 1u8; }; 2");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
}

/// types §3-4: an array literal's elements are values entering its one
/// element slot, where values meet; `!` is the bottom, so the slot is the
/// other elements' type and the arrays meet.
#[test]
fn an_array_with_a_never_element_joins_its_sibling() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let names = fx.names.clone();
    let (checked, script) = fx.script("let x = if $c { [0, 1] } else { [0, return 2] }; 0");
    checked.admitted();
    let TyKind::Array(element, _, _) = checked.kind(binding(&checked, &script, &names, "x")) else {
        panic!("not an array");
    };
    assert!(is_int(&checked, *element, IntTy::I64));
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
    let array = fx.ty(TyKind::Array(i64_ty, LenTerm::Known(3), ()));
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
    assert!(is_int(&checked, *element, IntTy::I64));
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
    let option = fx.ty(TyKind::Option(i64_ty, ()));
    fx.context("o", option);
    let (checked, script) = fx.script("let x = @o?; Some(x)");
    checked.admitted();
    let TyKind::Option(payload, _) = checked.kind(tail(&checked, &script)) else {
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
    let data = fx.ty(TyKind::Object { fields: Box::new([(name, bool_ty)]), label: () });
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
    let TyKind::Option(payload, _) = checked.kind(tail(&checked, &script)) else {
        panic!("not an option");
    };
    assert_eq!(checked.kind(*payload), &TyKind::Bool);
}

/// types §2, patterns §2: a variant written with no enum's name is
/// `Option`'s or `Result`'s, and names its own payload: of a
/// `Result<i64, bool>`, `Ok(x)` binds the `i64` and `Err(e)` the `bool`;
/// `Ok(1)` and `Err(true)` make a `Result` of an integer and a `bool`.
#[test]
fn a_primitive_variant_names_its_own_payload() {
    let mut fx = Fixture::new();
    let (int, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    let result = fx.ty(TyKind::Result(int, bool_ty, ()));
    fx.context("r", result);
    let (checked, script) = fx.script("match @r { Ok(x) => x, Err(e) => if e { 1 } else { 0 }, }");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));

    let (checked, script) = Fixture::new().script("if true { Ok(1) } else { Err(true) }");
    checked.admitted();
    let TyKind::Result(ok, err, _) = checked.kind(tail(&checked, &script)) else {
        panic!("not a result");
    };
    assert!(matches!(checked.kind(*ok), TyKind::Int(_)));
    assert_eq!(checked.kind(*err), &TyKind::Bool);
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
    let first = QualifiedRef::qualified(fx.namespace(&["a"]), fx.name("f"));
    let second = QualifiedRef::qualified(fx.namespace(&["b"]), fx.name("f"));
    for (id, ty) in [(first, of_int), (second, of_bool)] {
        let scheme = Scheme {
            ty,
            param_names: fx.param_names(1),
            bounds: Box::new([]),
            impl_of: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
        };
        fx.raw.functions.push((id, scheme));
    }
    let (checked, script) = fx.script("f(true)");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::I64));
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(checked.typeck.reached[&site].declaration, second);
}

/// A fixture declaring `show(i64) -> String` and `both(bool, i64) ->
/// String`, which the units below read their inputs through.
fn unit_fixture() -> Fixture {
    let mut fx = Fixture::new();
    let (bool_ty, i64_ty, string) = (fx.ty(TyKind::Bool), fx.int(IntTy::I64), fx.ty(TyKind::String));
    let show = fx.pure(&[i64_ty], string);
    fx.declare("show", show, &[]);
    let both = fx.pure(&[bool_ty, i64_ty], string);
    fx.declare("both", both, &[]);
    fx
}

/// programs §1, §2: a call by name reaches the unit at the callee's whole
/// name, its interface instantiated there: each argument enters the
/// parameter it names, and the call is the unit's result.
#[test]
fn a_call_by_name_reaches_the_unit_at_its_whole_name() {
    let (checked, script, units) = unit_fixture().script_with_units(&[(&["ns"], "greet", "show($user)")], "ns::greet(user: 1)");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::String);
    let Some(Expr::FuncCall { id, func, .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    assert_eq!(checked.typeck.units[id], UnitCall { unit: units[0], arguments: Box::new([0]) });
    assert_eq!(checked.typeck.names[&func.id()], Name::Unit(units[0]));
}

/// programs: a bare name reaches host declarations only, never a unit, so
/// a call of one by name is of no unit.
#[test]
fn a_bare_name_called_by_name_reaches_no_unit() {
    let (checked, _, _) = unit_fixture().script_with_units(&[(&["ns"], "greet", "show($user)")], "greet(user: 1)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NamedArgumentsToNonUnit]);
}

/// programs §1, §2: a unit's parameters are passed by name, so an argument
/// passed to one by position is refused.
#[test]
fn a_unit_called_by_position_is_refused() {
    let (checked, _, units) = unit_fixture().script_with_units(&[(&["ns"], "greet", "show($user)")], "ns::greet(1)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::UnitCalledByPosition { unit: units[0] }]);
}

/// programs: named arguments reach a unit only; a host declaration's
/// parameters are passed by position.
#[test]
fn a_declaration_called_by_name_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    let ns = fx.namespace(&["ns"]);
    fx.declare_at(ns, "f", ty);
    let (checked, _) = fx.script_calling(&Interfaces::new(), "ns::f(p0: 1)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NamedArgumentsToNonUnit]);
}

/// programs §1, §2: arguments are matched to the unit's parameters by
/// name, in any order written, and passed in the parameters' order, the
/// order of the inputs' names (`times` before `user`).
#[test]
fn arguments_by_name_are_matched_to_the_parameters_in_any_order() {
    let (checked, script, units) =
        unit_fixture().script_with_units(&[(&["ns"], "greet", "both($user, $times)")], "ns::greet(user: true, times: 2)");
    checked.admitted();
    let Some(Expr::FuncCall { id, .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    assert_eq!(checked.typeck.units[id], UnitCall { unit: units[0], arguments: Box::new([1, 0]) });
}

/// programs §1, §2: every parameter is given exactly once and every name
/// is a parameter's; each mismatch is refused as what it is, naming what
/// the program wrote or left out.
#[test]
fn a_missing_duplicated_or_unknown_argument_name_is_refused() {
    let cases: [(&str, fn(QualifiedRef, &Interner) -> RefusalKind); 3] = [
        ("ns::greet(user: true)", |unit, names| RefusalKind::MissingArgument { unit, name: names.intern("times") }),
        ("ns::greet(user: true, user: false, times: 1)", |unit, names| RefusalKind::DuplicateArgument { unit, name: names.intern("user") }),
        ("ns::greet(user: true, times: 1, name: 2)", |unit, names| RefusalKind::UnknownArgument { unit, name: names.intern("name") }),
    ];
    for (source, refused) in cases {
        let fx = unit_fixture();
        let names = fx.names.clone();
        let (checked, _, units) = fx.script_with_units(&[(&["ns"], "greet", "both($user, $times)")], source);
        assert_eq!(checked.refusals(), vec![&refused(units[0], &names)], "{source}");
    }
}

/// programs: a unit with no inputs is called with no argument, which is
/// passed neither by position nor by name.
#[test]
fn a_unit_of_no_parameter_is_called_with_none() {
    let (checked, script, units) = unit_fixture().script_with_units(&[(&["ns"], "u", "true")], "ns::u()");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    let Some(Expr::FuncCall { id, .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    assert_eq!(checked.typeck.units[id], UnitCall { unit: units[0], arguments: Box::new([]) });
}

/// functions §2, programs §1-§2: a unit is generalized as a let-bound
/// function value is. `$x + 1` settles nothing of `x`'s width, so its
/// scheme keeps `x` open with the addition as a requirement, and two
/// callers instantiate it at two widths.
#[test]
fn a_unit_is_generalized_and_instantiated_at_each_call() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::U8), TyKind::Int(IntTy::I64)]);
    let (checked, script, _) = fx.script_with_units(&[(&["ns"], "inc", "$x + 1")], "(ns::inc(x: 1u8), ns::inc(x: 2))");
    checked.admitted();
    let TyKind::Tuple(parts, ()) = checked.kind(tail(&checked, &script)) else {
        panic!("the tail is a tuple");
    };
    assert_eq!(checked.kind(parts[0]), &TyKind::Int(IntTy::U8));
    assert_eq!(checked.kind(parts[1]), &TyKind::Int(IntTy::I64));
}

/// functions §2: a requirement the unit could not decide is made at each
/// call, so a call at a type no declaration answers is refused there.
#[test]
fn a_unit_requirement_no_declaration_answers_is_refused_at_the_call() {
    let mut fx = Fixture::new();
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::I64)]);
    let (checked, _, _) = fx.script_with_units(&[(&["ns"], "inc", "$x + 1")], "ns::inc(x: true)");
    assert!(!checked.refusals().is_empty(), "adding to a bool is refused at the call");
}

/// programs: a function stands at a name of any depth, and a qualified
/// name reaches the declaration at its whole path.
#[test]
fn a_three_segment_callee_reaches_the_declaration_at_its_whole_path() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    let a_b = fx.namespace(&["a", "b"]);
    let f = fx.declare_at(a_b, "f", ty);
    let (checked, script) = fx.script("a::b::f(1)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(checked.typeck.reached[&site].declaration, f);
}

/// Two segments are a callee's namespace of one segment where a function
/// is declared there, and otherwise the variant `E::Tag`.
#[test]
fn a_two_segment_callee_reaches_a_declaration_or_is_a_variant() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    let f = fx.overload("ns", "f", ty);
    let enum_name = QualifiedRef::root(fx.name("E"));
    let tag = fx.name("Tag");
    let names = fx.names.clone();
    let (checked, script) = fx.script("let v = E::Tag(1); ns::f(2)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(checked.typeck.reached[&site].declaration, f);
    let TyKind::Enum { name, variants, .. } = checked.kind(binding(&checked, &script, &names, "v")) else {
        panic!("`E::Tag(1)` is a variant of the enum `E`");
    };
    assert_eq!(*name, enum_name);
    assert!(variants.iter().any(|(written, payload)| *written == tag && payload.is_some()));
}

/// A qualified name compares its whole path: `a::b::f` reaches neither
/// `b::f` nor `a::f`, and a name deeper than two segments is no variant, so
/// it is unknown.
#[test]
fn a_three_segment_callee_reaches_no_declaration_at_part_of_its_path() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    fx.overload("b", "f", ty);
    fx.overload("a", "f", ty);
    let f = fx.name("f");
    let (checked, _) = fx.script("a::b::f(1)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::UnknownName { name: f }]);
}

/// types §6: a bare name reaches every declaration of that name, whatever
/// its namespace, however deep.
#[test]
fn a_bare_name_reaches_a_declaration_at_a_deep_namespace() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    let a_b = fx.namespace(&["a", "b"]);
    let f = fx.declare_at(a_b, "f", ty);
    let (checked, script) = fx.script("f(1)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    assert_eq!(checked.typeck.reached[&site].declaration, f);
}

/// types §6: several declarations left is a refusal.
#[test]
fn a_call_two_declarations_answer_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let ty = fx.pure(&[i64_ty], i64_ty);
    let f = fx.name("f");
    for namespace in ["a", "b"] {
        let id = QualifiedRef::qualified(fx.namespace(&[namespace]), f);
        let scheme = Scheme {
            ty,
            param_names: fx.param_names(1),
            bounds: Box::new([]),
            impl_of: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            awaits: false,
            specialized: Box::new([]),
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
    let option_t = fx.ty(TyKind::Option(t, ()));
    let at_option = fx.function(&[option_t], t, EffectTerm::Known(crate::ty::Effect::pure()));
    fx.lang_item(unwrap, &[at_option]);
    let bool_ty = fx.ty(TyKind::Bool);
    let option = fx.ty(TyKind::Option(bool_ty, ()));
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
            binder: CaptureOf::Binding(let_binder(&script, 0)),
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
    let array = fx.ty(TyKind::Array(bool_ty, LenTerm::Known(2), ()));
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
    let (checked, script) = fx.script("let y = true; let f = move |x| -> y; f(1u8)");
    checked.admitted();
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    assert_eq!(
        &closure.captures[..],
        &[Captured {
            name: y,
            binder: CaptureOf::Binding(let_binder(&script, 0)),
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

/// functions §3, D93: a `move` lambda that moves its capture out is called
/// once, and every copy of its type says so: the lambda's own, the binding
/// it enters, a binding that value is copied into, and the callee at a call
/// through that binding. Settling the captures after the fixpoint binds the
/// lambda's `calls` variable, which the copies share by link (D14 had each
/// slot keep a stale copy).
#[test]
fn a_move_lambda_that_moves_its_capture_out_is_once_in_every_copy() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    fx.context("s", string);
    let names = fx.names.clone();
    let (checked, script) = fx.script("let s = @s; let f = move |x| -> s; let g = f; g(1)");
    checked.admitted();
    let Stmt::LetBind { expr: lambda, .. } = &script.stmts[1] else {
        panic!("not a let");
    };
    let Some(Expr::FuncCall { func: callee, .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    let copies = [
        ("the lambda", checked.typeck.types[&lambda.id()]),
        ("f", binding(&checked, &script, &names, "f")),
        ("g", binding(&checked, &script, &names, "g")),
        ("the callee", checked.typeck.types[&callee.id()]),
    ];
    for (copy, ty) in copies {
        let TyKind::Fn { calls, holds, .. } = checked.kind(ty) else {
            panic!("{copy} is a function");
        };
        assert_eq!((*calls, *holds), (Calls::Once, Holds::Nothing), "{copy}");
    }
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    assert_eq!(closure.calls, Calls::Once);
}

/// types §2, D93: two lambdas meeting in one slot join there, so the slot
/// and both lambdas are once where one is, and hold the stronger loan.
#[test]
fn lambdas_joined_in_one_slot_share_their_holds_and_calls() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    let bool_ty = fx.ty(TyKind::Bool);
    let i64_ty = fx.int(IntTy::I64);
    fx.context("s", string);
    fx.context("n", i64_ty);
    fx.input("c", bool_ty);
    let names = fx.names.clone();
    let (checked, script) =
        fx.script("let s = @s; let r = &mut @n; let f = if $c { move |x| -> { let t = s; x } } else { |x| -> { *r = x; x } }; 1");
    checked.admitted();
    let f = binding(&checked, &script, &names, "f");
    let lambdas: Vec<Ty> = checked.typeck.closures.keys().map(|lambda| checked.typeck.types[lambda]).collect();
    assert_eq!(lambdas.len(), 2);
    for ty in lambdas.into_iter().chain([f]) {
        let TyKind::Fn { calls, holds, .. } = checked.kind(ty) else {
            panic!("a function");
        };
        assert_eq!((*calls, *holds), (Calls::Once, Holds::Mut));
    }
}

/// ownership §5, D93: a `move` lambda holds what the values it moved in
/// hold. `f` moves `h` in before `h` is assigned a later lambda that holds
/// `r` exclusively; the two lambdas share `h`'s type, so `f` holds `Mut`
/// too, though the later lambda settles after `f`.
#[test]
fn a_move_lambda_holds_what_a_later_lambda_sharing_its_captures_type_holds() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let names = fx.names.clone();
    let (checked, script) =
        fx.script("let r = &mut @n; let h = |x| -> x; let f = move |x| -> h(x); h = |x| -> { *r = x; x }; f(1)");
    checked.admitted();
    for name in ["h", "f"] {
        let TyKind::Fn { holds, .. } = checked.kind(binding(&checked, &script, &names, name)) else {
            panic!("`{name}` is a function");
        };
        assert_eq!(*holds, Holds::Mut, "`{name}`");
    }
}

/// `apply(f: fn(i64) -> i64) -> i64`, whose closure parameter is pure,
/// holds nothing and is called many times, as declared.
fn declare_apply(fx: &mut Fixture) {
    let i64_ty = fx.int(IntTy::I64);
    let closure = fx.pure(&[i64_ty], i64_ty);
    let apply = fx.pure(&[closure], i64_ty);
    fx.declare("apply", apply, &[]);
}

/// D94, ownership §5: a slot declared `holds: Nothing` is made equal to
/// the lambda entering it, so a lambda that settles to holding `r`
/// exclusively is refused at the lambda, naming both.
#[test]
fn a_mut_lambda_entering_a_slot_that_holds_nothing_is_refused() {
    let mut fx = Fixture::new();
    declare_apply(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let (checked, script) = fx.script("let r = &mut @n; apply(|x| -> { *r = x; x })");
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    assert_eq!(
        checked.refusals(),
        vec![&RefusalKind::ClosureExceedsSlot(Exceeding::Holds { needs: Holds::Mut, slot: Holds::Nothing })]
    );
    assert_eq!(checked.typeck.refusals[0].span, args[0].span(), "refused at the lambda");
}

/// D94, functions §3: a closure parameter declared many takes no lambda
/// that moves its capture out, called once.
#[test]
fn a_once_lambda_passed_to_a_closure_parameter_declared_many_is_refused() {
    let mut fx = Fixture::new();
    declare_apply(&mut fx);
    let string = fx.ty(TyKind::String);
    fx.context("s", string);
    let (checked, _) = fx.script("let s = @s; apply(move |x| -> { let t = s; x })");
    assert_eq!(
        checked.refusals(),
        vec![&RefusalKind::ClosureExceedsSlot(Exceeding::Calls { needs: Calls::Once, slot: Calls::Many })]
    );
}

/// D94, effects: the closure parameter is declared pure. A lambda whose
/// body's effect is opaque already when it enters is a mismatch of the two
/// effects there. One whose effect grows only after it entered, since its
/// call's declaration waits on `x`'s type, which the slot decides, is
/// refused at the lambda rather than joined past the slot.
#[test]
fn a_lambda_whose_effect_grows_past_a_pure_slot_is_refused() {
    let mut fx = Fixture::new();
    declare_apply(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    let bool_ty = fx.ty(TyKind::Bool);
    let opaque = |fx: &mut Fixture, param| fx.function(&[param], param, EffectTerm::Known(crate::ty::Effect::opaque()));
    let at_int = opaque(&mut fx, i64_ty);
    let at_bool = opaque(&mut fx, bool_ty);
    fx.overload("ints", "touch", at_int);
    fx.overload("bools", "touch", at_bool);
    let (checked, script) = fx.script("apply(|x| -> touch(x))");
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    assert_eq!(
        checked.refusals(),
        vec![&RefusalKind::ClosureExceedsSlot(Exceeding::Effect {
            needs: crate::ty::Effect::opaque(),
            slot: crate::ty::Effect::pure()
        })]
    );
    assert_eq!(checked.typeck.refusals[0].span, args[0].span(), "refused at the lambda");
}

/// D96, types §6: a lambda whose effect already exceeds the pure slot when
/// it enters is refused with the same kind, at the same place, as one
/// whose effect grows past the slot later: the outcome does not depend on
/// the order of checking.
#[test]
fn a_lambda_whose_effect_exceeds_a_pure_slot_on_entering_is_refused_as_when_it_grows_past() {
    let mut fx = Fixture::new();
    declare_apply(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    let opaque = fx.function(&[i64_ty], i64_ty, EffectTerm::Known(crate::ty::Effect::opaque()));
    fx.declare("touch", opaque, &[]);
    let (checked, script) = fx.script("apply(|x| -> touch(x))");
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("the tail is a call");
    };
    assert_eq!(
        checked.refusals(),
        vec![&RefusalKind::ClosureExceedsSlot(Exceeding::Effect {
            needs: crate::ty::Effect::opaque(),
            slot: crate::ty::Effect::pure()
        })]
    );
    assert_eq!(checked.typeck.refusals[0].span, args[0].span(), "refused at the lambda");
}

/// D96: a function value that is no lambda, whose effect does not fit the
/// function type it enters, is a mismatch naming the two effects. A
/// declaration is no function value (extern-fn §1); a call's result is.
#[test]
fn a_function_value_a_call_gives_whose_effect_does_not_fit_is_a_mismatch_of_effects() {
    let mut fx = Fixture::new();
    declare_apply(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    let opaque = fx.function(&[i64_ty], i64_ty, EffectTerm::Known(crate::ty::Effect::opaque()));
    let touching = fx.pure(&[], opaque);
    fx.declare("touching", touching, &[]);
    let (checked, _) = fx.script("apply(touching())");
    let [RefusalKind::Mismatch(FrozenMismatch { mismatch: Mismatch::Effects { a_is, b_is, .. }, .. })] = checked.refusals()[..] else {
        panic!("one mismatch of effects: {:?}", checked.refusals());
    };
    let mut effects = [a_is.clone(), b_is.clone()];
    effects.sort_by_key(|effect| effect.reissue());
    assert_eq!(effects, [crate::ty::Effect::pure(), crate::ty::Effect::opaque()]);
}

/// A refusal states the effects that did not meet, each as the type printer
/// writes an effect, frozen at writeback: the declared slot's
/// `(pure, no_suspend)` and the call's result's `opaque`.
#[test]
fn a_mismatch_of_effects_prints_both_effects() {
    let mut fx = Fixture::new();
    declare_apply(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    let opaque = fx.function(&[i64_ty], i64_ty, EffectTerm::Known(crate::ty::Effect::opaque()));
    let touching = fx.pure(&[], opaque);
    fx.declare("touching", touching, &[]);
    let names = fx.names.clone();
    let (checked, _) = fx.script("apply(touching())");
    let [refusal] = &checked.typeck.refusals[..] else {
        panic!("one refusal: {:?}", checked.refusals());
    };
    let written = refusal.kind.written(&names);
    let shown = written.strip_prefix("Mismatch Effects ").unwrap_or_else(|| panic!("{written}"));
    assert!(
        shown == "(pure, no_suspend) opaque" || shown == "opaque (pure, no_suspend)",
        "{written}"
    );
}

/// `apply`, a pure `core::add` at `i64`, and `touch(i64) -> i64`, opaque.
fn pure_add_and_opaque_touch(fx: &mut Fixture) {
    declare_apply(fx);
    let add = fx.raw.core.add;
    fx.arithmetic(add, &[TyKind::Int(IntTy::I64)]);
    let i64_ty = fx.int(IntTy::I64);
    let opaque = fx.function(&[i64_ty], i64_ty, EffectTerm::Known(crate::ty::Effect::opaque()));
    fx.declare("touch", opaque, &[]);
}

/// The effect of the lambda at `let` statement `at`.
fn lambda_effect(checked: &Checked, script: &Script, at: usize) -> crate::ty::Effect {
    let Stmt::LetBind { expr, .. } = &script.stmts[at] else {
        panic!("statement {at} is not a `let`");
    };
    let TyKind::Fn { effect, .. } = checked.kind(checked.typeck.types[&expr.id()]) else {
        panic!("statement {at} binds a lambda");
    };
    match effect {
        EffectTerm::Known(effect) => effect.clone(),
        EffectTerm::Var(var) => match *var {},
    }
}

/// D94: a lambda within what its slot states is admitted: a pure lambda
/// that captures nothing and copies what it reads. Its `+` reaches a pure
/// impl; the generic `core::add` alone would leave its effect to the
/// default, opaque (types §5).
#[test]
fn a_lambda_within_its_slot_is_admitted() {
    let mut fx = Fixture::new();
    pure_add_and_opaque_touch(&mut fx);
    let (checked, _) = fx.script("apply(|x| -> 1)");
    checked.admitted();
}

/// D95, effects §1: a body's effect is the join of its calls', one way. A
/// pure lambda passed into a pure slot and also called in an opaque body
/// stays pure: the body's effect never flows back into the callee.
#[test]
fn a_pure_lambda_called_in_an_opaque_body_stays_pure() {
    let mut fx = Fixture::new();
    pure_add_and_opaque_touch(&mut fx);
    let (checked, script) = fx.script("let f = |x| -> x + 1; let a = apply(f); let b = f(1); touch(b)");
    checked.admitted();
    assert_eq!(lambda_effect(&checked, &script, 0), crate::ty::Effect::pure());
    assert_eq!(checked.typeck.body_effect, crate::ty::Effect::opaque());
}

/// D95, effects §1: effects flow one way along a chain of calls. `g`
/// calls `f`, which is opaque, and `p`, which is pure; the body calls `g`.
/// `f`'s effect reaches `g` and the body; nothing reaches `p` or `f`
/// back.
#[test]
fn effects_flow_one_way_along_a_chain_of_calls() {
    let mut fx = Fixture::new();
    pure_add_and_opaque_touch(&mut fx);
    let (checked, script) =
        fx.script("let f = |x| -> touch(x); let p = |x| -> x + 1; let g = |x| -> p(f(x)); g(1)");
    checked.admitted();
    let opaque = crate::ty::Effect::opaque();
    assert_eq!(lambda_effect(&checked, &script, 0), opaque, "f");
    assert_eq!(lambda_effect(&checked, &script, 1), crate::ty::Effect::pure(), "p");
    assert_eq!(lambda_effect(&checked, &script, 2), opaque, "g");
    assert_eq!(checked.typeck.body_effect, opaque, "the body");
}

/// types §5, D93: a function type no lambda claims and nothing decides
/// holds no loan and may be called many times: here, the parameter a
/// lambda only calls.
#[test]
fn an_unclaimed_function_types_holds_and_calls_take_the_defaults() {
    let fx = Fixture::new();
    let (checked, script) = fx.script("let h = |g| -> g(1); 1");
    checked.admitted();
    let Stmt::LetBind {
        expr: Expr::Lambda { params, .. },
        ..
    } = &script.stmts[0]
    else {
        panic!("a lambda bound by `let`");
    };
    let TyKind::Fn { calls, holds, .. } = checked.kind(checked.typeck.types[&params[0].id]) else {
        panic!("`g` is called, so it is a function");
    };
    assert_eq!((*calls, *holds), (Calls::Many, Holds::Nothing));
}

/// types §6: a tuple is reached by its parts in order, past its head; a
/// declaration whose tuple parameter is of another arity is left at the
/// head, and its parts are not read. With `p` over `((i64, i64, i64))` and
/// `((u8, u8))`, each call reaches the one of its arity.
#[test]
fn a_tuple_argument_reaches_past_a_parameter_of_another_arity() {
    let fixture = || {
        let mut fx = Fixture::new();
        let (i64_ty, u8_ty) = (fx.int(IntTy::I64), fx.int(IntTy::U8));
        let three = fx.ty(TyKind::Tuple(Box::new([i64_ty, i64_ty, i64_ty]), ()));
        let two = fx.ty(TyKind::Tuple(Box::new([u8_ty, u8_ty]), ()));
        let (of_three, of_two) = (fx.pure(&[three], i64_ty), fx.pure(&[two], i64_ty));
        let by_three = fx.overload("n0", "p", of_three);
        let by_two = fx.overload("n1", "p", of_two);
        (fx, by_three, by_two)
    };
    for (source, three) in [("p((1i64, 2i64, 3i64))", true), ("p((1u8, 2u8))", false)] {
        let (fx, by_three, by_two) = fixture();
        let (checked, _) = fx.script(source);
        checked.admitted();
        let wanted = if three { by_three } else { by_two };
        let reached: Vec<QualifiedRef> = checked.typeck.reached.values().map(|chosen| chosen.declaration).collect();
        assert_eq!(reached, vec![wanted], "{source}");
    }
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

/// control-flow: `break` and `continue` mean what they mean in Rust, which
/// refuses one with no label in a `while`'s condition or a `while let`'s
/// source, with a loop outside or none; there are no labels. One in a loop
/// inside the condition leaves that loop, and one in a `for`'s head, which
/// is evaluated once before its loop, leaves the loop outside.
#[test]
fn break_or_continue_in_a_while_condition_is_refused() {
    let fixture = || {
        let mut fx = Fixture::new();
        let bool_ty = fx.ty(TyKind::Bool);
        fx.input("c", bool_ty);
        fx
    };
    for source in [
        "while $c { while (if $c { break; } else { true }) { } } 0",
        "while $c { while (if $c { continue; } else { true }) { } } 0",
        "while $c { while let Some(x) = (if $c { break; } else { Some(1) }) { } } 0",
        "while (if $c { break; } else { true }) { } 0",
    ] {
        let (checked, _) = fixture().script(source);
        assert_eq!(checked.refusals(), vec![&RefusalKind::InLoopCondition], "{source}");
    }
    for source in [
        "while $c { while (if $c { while $c { break; } true } else { false }) { } } 0",
        "while $c { for x in (if $c { break; } else { [1, 2] }) { } } 0",
    ] {
        let (checked, _) = fixture().script(source);
        assert!(checked.refusals().is_empty(), "{source}: {:?}", checked.refusals());
    }
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
    let Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. } = script.tail.as_deref().unwrap() else {
        panic!("not a call");
    };
    let steps = checked.steps(args[0].id());
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].via.declaration, as_slice);
    let TyKind::Ref(Mutability::Shared, viewed) = checked.kind(steps[0].to) else {
        panic!("a view is a reference");
    };
    assert!(matches!(checked.kind(*viewed), TyKind::Slice(_)));
}

// Comparisons (operators §4)

/// operators §2: `core` implements nothing, and where no impl answers the
/// program is refused. With no `core::clone` impl declared anywhere, a
/// `String`'s `.clone()` has none to reach.
#[test]
fn a_lang_item_given_no_impls_answers_no_reach() {
    let mut fx = Fixture::new();
    let clone = fx.raw.core.clone;
    let string = fx.ty(TyKind::String);
    fx.input("s", string);
    let (checked, _) = fx.script("$s.clone()");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: clone.name }]);
}

/// types §1: an array's `clone` is lifted from its element's, so a
/// `[String; 2]`'s `.clone()` answers where `String` has a `clone` impl, and
/// is refused at the element where it has none.
#[test]
fn a_clone_of_an_array_lifts_from_its_elements_clone() {
    for clones_strings in [false, true] {
        let mut fx = Fixture::new();
        let clone = fx.raw.core.clone;
        let string = fx.ty(TyKind::String);
        if clones_strings {
            fx.clones(&[string]);
        }
        let array = fx.ty(TyKind::Array(string, crate::ty::LenTerm::Known(2), ()));
        fx.input("a", array);
        let (checked, _) = fx.script("$a.clone()");
        match clones_strings {
            true => checked.admitted(),
            false => assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: clone.name }]),
        }
    }
}

/// types §1: a tuple's `clone` is lifted from its parts', so its `String`
/// part, which no impl answers here, is refused, and its `i64` part, which
/// one does, is not.
#[test]
fn a_tuple_clone_with_a_part_no_impl_clones_is_refused_at_the_part() {
    let mut fx = Fixture::new();
    let clone = fx.raw.core.clone;
    let (string, i64_ty) = (fx.ty(TyKind::String), fx.int(IntTy::I64));
    fx.clones(&[i64_ty]);
    let pair = fx.ty(TyKind::Tuple(Box::new([string, i64_ty]), ()));
    fx.input("t", pair);
    let (checked, _) = fx.script("$t.clone()");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: clone.name }]);
}

/// types §1: `clone` maps one to one onto a structure's parts, so a value
/// of each of the six transparent structured types, nested in one another,
/// clones by a lifted impl whose every leaf is its part's own impl.
#[test]
fn a_clone_lifts_over_every_transparent_structured_type_nested() {
    let mut fx = Fixture::new();
    let clone = fx.raw.core.clone;
    let (string, i64_ty, bool_ty) = (fx.ty(TyKind::String), fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    fx.clones(&[string, i64_ty, bool_ty]);
    let (a, b, red, blue) = (fx.name("a"), fx.name("b"), fx.name("Red"), fx.name("Blue"));
    let color = fx.ty(TyKind::Enum {
        name: QualifiedRef::root(fx.name("Color")),
        variants: Box::new([(blue, None), (red, Some(string))]),
        label: (),
    });
    let option = fx.ty(TyKind::Option(i64_ty, ()));
    let result = fx.ty(TyKind::Result(string, bool_ty, ()));
    let array = fx.ty(TyKind::Array(option, LenTerm::Known(2), ()));
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, array), (b, result)]), label: () });
    let whole = fx.ty(TyKind::Tuple(Box::new([object, color]), ()));
    fx.input("t", whole);
    let (checked, script) = fx.script("$t.clone()");
    checked.admitted();
    let Some(Expr::MethodCall { callee_id, .. }) = script.tail.as_deref() else {
        panic!("not a method call");
    };
    assert_eq!(checked.typeck.reached[callee_id].declaration, clone);
    assert_eq!(checked.typeck.reached[callee_id].r#impl, Some(Impl::Structural));
    let (mut leaves, mut copies) = (Vec::new(), Vec::new());
    let mut open = vec![&checked.typeck.structure_trees[callee_id]];
    while let Some(tree) = open.pop() {
        match tree {
            super::StructureTree::Parts(parts) => open.extend(parts.iter().map(|(_, part)| part)),
            super::StructureTree::Reached { chosen, .. } => leaves.push((chosen.declaration, chosen.r#impl)),
            super::StructureTree::Own(own) => copies.push(*own),
        }
    }
    // The impls: the `Result`'s `String` payload and `Red`'s. The array of
    // `Option<i64>`, which copies, and the `Result`'s `bool` payload are
    // their copies (ownership §2).
    assert_eq!(leaves.len(), 2);
    assert!(leaves.iter().all(|(declaration, r#impl)| *declaration == clone && matches!(r#impl, Some(Impl::Concrete(_)))));
    assert_eq!(copies, vec![super::Own::Copy; 2]);
}

/// types §1: a structure has a lifted `clone` only where every part has an
/// impl; one part of none, however deep, refuses it there. A part that
/// copies needs none (ownership §2).
#[test]
fn a_clone_of_a_structure_with_a_part_of_no_impl_is_refused() {
    let mut fx = Fixture::new();
    let clone = fx.raw.core.clone;
    let (string, bool_ty) = (fx.ty(TyKind::String), fx.ty(TyKind::Bool));
    let a = fx.name("a");
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, string)]), label: () });
    let option = fx.ty(TyKind::Option(object, ()));
    let whole = fx.ty(TyKind::Tuple(Box::new([bool_ty, option]), ()));
    fx.input("t", whole);
    let (checked, _) = fx.script("$t.clone()");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: clone.name }]);
}

/// types §1: `eq` combines its parts' answers, so a tuple has a lifted `eq`
/// only where every part's `eq` is declared an equivalence.
#[test]
fn eq_lifts_only_where_every_parts_eq_is_declared_an_equivalence() {
    for states_equivalence in [false, true] {
        let mut fx = Fixture::new();
        let eq = fx.raw.core.eq;
        let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
        let shared = fx.reference(Mutability::Shared, i64_ty);
        let at_int = fx.pure(&[shared, shared], bool_ty);
        let access: Box<[crate::decl::declared::Access]> = match states_equivalence {
            true => Box::new([crate::decl::declared::Access::Equivalence]),
            false => Box::new([]),
        };
        fx.lang_item_stating(eq, &[at_int], DeclaredFacts { access, ..Default::default() });
        let pair = fx.ty(TyKind::Tuple(Box::new([i64_ty, i64_ty]), ()));
        fx.input("t", pair);
        let (checked, script) = fx.script("eq(&$t, &$t)");
        match states_equivalence {
            true => {
                checked.admitted();
                let Some(Expr::FuncCall { id, .. }) = script.tail.as_deref() else {
                    panic!("not a call");
                };
                assert_eq!(checked.typeck.reached[id].r#impl, Some(Impl::Structural));
            }
            false => assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: eq.name }; 2]),
        }
    }
}

// Structured coercion (types §1, §3)

/// types §1, §3: an object entering a declaration's parameter narrows to
/// the fields the parameter names; the others end where it is narrowed.
#[test]
fn an_object_narrows_to_the_fields_a_declarations_parameter_names() {
    let mut fx = Fixture::new();
    let (a, b) = (fx.name("a"), fx.name("b"));
    let (i64_ty, unit) = (fx.int(IntTy::I64), fx.ty(TyKind::Unit));
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, i64_ty)]), label: () });
    let takes = fx.pure(&[object], unit);
    fx.declare("f", takes, &[]);
    let (checked, script) = fx.script("f({ a: 1, b: true, })");
    checked.admitted();
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("not a call");
    };
    let super::Coercion::Structured { shape, through: None, to, parts } = &checked.typeck.coercions[&args[0].id()] else {
        panic!("an object is coerced by its shape");
    };
    assert_eq!(shape, &ShapeCoercion::Narrowed { left: Box::new([b]) });
    let TyKind::Object { fields, label: () } = checked.kind(*to) else {
        panic!("coerced to the parameter's object");
    };
    assert_eq!(fields.iter().map(|(name, _)| *name).collect::<Vec<_>>(), vec![a]);
    assert_eq!(parts.iter().map(|(key, _)| *key).collect::<Vec<_>>(), vec![PartKey::Field(a)]);
}

/// types §3: an object narrows only to fields it holds.
#[test]
fn an_object_lacking_a_field_its_parameter_names_is_refused() {
    let mut fx = Fixture::new();
    let (a, b) = (fx.name("a"), fx.name("b"));
    let (i64_ty, unit) = (fx.int(IntTy::I64), fx.ty(TyKind::Unit));
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, i64_ty), (b, i64_ty)]), label: () });
    let takes = fx.pure(&[object], unit);
    fx.declare("f", takes, &[]);
    let (checked, _) = fx.script("f({ a: 1, })");
    assert!(matches!(checked.refusals()[..], [RefusalKind::Mismatch(FrozenMismatch { mismatch: Mismatch::Heads { .. }, .. })]), "{:?}", checked.refusals());
}

/// A refusal states the types that did not meet: a heads mismatch prints
/// the value's type and the slot's after its shape, frozen at writeback,
/// where the literal no use decided is `i64` (types §5). A refusal cloned
/// is equal to it, its types in a table of its own.
#[test]
fn a_mismatch_of_heads_prints_both_types() {
    let mut fx = Fixture::new();
    let (a, b) = (fx.name("a"), fx.name("b"));
    let (i64_ty, unit) = (fx.int(IntTy::I64), fx.ty(TyKind::Unit));
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, i64_ty), (b, i64_ty)]), label: () });
    let takes = fx.pure(&[object], unit);
    fx.declare("f", takes, &[]);
    let names = fx.names.clone();
    let (checked, _) = fx.script("f({ a: 1, })");
    let [refusal] = &checked.typeck.refusals[..] else {
        panic!("one refusal: {:?}", checked.refusals());
    };
    assert_eq!(refusal.kind.written(&names), "Mismatch Heads {a: i64} {a: i64, b: i64}");
    assert_eq!(&refusal.clone(), refusal);
}

/// types §3: an enum entering a declaration's parameter widens to the
/// variants the parameter names, and one holding a variant it does not
/// name is refused.
#[test]
fn an_enum_widens_to_the_variants_a_declarations_parameter_names() {
    for (source, admitted) in [("f(Color::Red)", true), ("f(if true { Color::Red } else { Color::Green })", false)] {
        let mut fx = Fixture::new();
        let (red, blue) = (fx.name("Red"), fx.name("Blue"));
        let unit = fx.ty(TyKind::Unit);
        let mut variants = [(blue, None), (red, None)];
        variants.sort_by_key(|(tag, _)| *tag);
        let color = fx.ty(TyKind::Enum { name: QualifiedRef::root(fx.name("Color")), variants: Box::new(variants), label: () });
        let takes = fx.pure(&[color], unit);
        fx.declare("f", takes, &[]);
        let (checked, script) = fx.script(source);
        if !admitted {
            assert!(matches!(checked.refusals()[..], [RefusalKind::Mismatch(FrozenMismatch { mismatch: Mismatch::Heads { .. }, .. })]), "{:?}", checked.refusals());
            continue;
        }
        checked.admitted();
        let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
            panic!("not a call");
        };
        let super::Coercion::Structured { shape, .. } = &checked.typeck.coercions[&args[0].id()] else {
            panic!("an enum is coerced by its shape");
        };
        assert_eq!(shape, &ShapeCoercion::Widened { taken: Box::new([blue]) });
    }
}

/// types §3: a tuple entering a declaration's parameter keeps its arity,
/// and each part is coerced up the cast tree as any value entering a slot.
/// The same two tuples meeting in one body's slot are not coerced part by
/// part: a constructor is invariant (types §4), so they are refused.
#[test]
fn a_tuple_part_casts_at_a_declarations_parameter_but_not_where_two_meet_in_a_body() {
    let widening = || {
        let mut fx = Fixture::new();
        let (u8_ty, i64_ty, bool_ty, unit) = (fx.int(IntTy::U8), fx.int(IntTy::I64), fx.ty(TyKind::Bool), fx.ty(TyKind::Unit));
        let widening = fx.pure(&[u8_ty], i64_ty);
        let widen = fx.declare("widen", widening, &[]);
        fx.raw.casts.push(CastRule { from: u8_ty, to: i64_ty, via: widen });
        let pair = fx.ty(TyKind::Tuple(Box::new([i64_ty, bool_ty]), ()));
        let takes = fx.pure(&[pair], unit);
        fx.declare("f", takes, &[]);
        (fx, widen)
    };
    let (fx, widen) = widening();
    let (checked, script) = fx.script("f((1u8, true))");
    checked.admitted();
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("not a call");
    };
    let super::Coercion::Structured { shape, parts, .. } = &checked.typeck.coercions[&args[0].id()] else {
        panic!("a tuple is coerced by its shape");
    };
    assert_eq!(shape, &ShapeCoercion::Kept);
    let [(PartKey::Index(0), Some(super::Coercion::Steps(steps))), (PartKey::Index(1), None)] = &parts[..] else {
        panic!("the first part is coerced by steps, the second enters as it is: {parts:?}");
    };
    assert_eq!(steps[0].via.declaration, widen);

    let (fx, _) = widening();
    let (checked, _) = fx.script("if true { (1u8, true) } else { (2i64, false) }");
    assert!(matches!(checked.refusals()[..], [RefusalKind::Mismatch(_)]), "{:?}", checked.refusals());
}

// Forms decided by their operand's head (types §6)

/// types §6: a field read through a reference is decided once the object's
/// type is known, which here is only after the lambda is called.
#[test]
fn a_field_through_a_reference_waits_for_its_objects_type() {
    let mut fx = Fixture::new();
    let a = fx.name("a");
    let bool_ty = fx.ty(TyKind::Bool);
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, bool_ty)]), label: () });
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
    assert!(is_int(&checked, *target, IntTy::U8));
}

/// operators §3: an index is decided by its container's type, known here
/// only once the lambda is called.
#[test]
fn an_index_waits_for_its_containers_type() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    let array = fx.ty(TyKind::Array(bool_ty, LenTerm::Known(2), ()));
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

/// operators §2: `^` is the language's own on `bool` and the integers,
/// and on no other type.
#[test]
fn xor_is_the_languages_own_on_bool_and_the_integers() {
    let (checked, script) = Fixture::new().script("true ^ false");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    assert!(checked.typeck.reached.is_empty());
    let (checked, script) = Fixture::new().script("1u8 ^ 2");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
    let (checked, _) = Fixture::new().script("(1, 2) ^ (3, 4)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NotBoolOrInteger]);
}

/// operators §2: `!` is the language's own on `bool` alone.
#[test]
fn not_is_the_languages_own_on_bool() {
    let (checked, script) = Fixture::new().script("!true");
    checked.admitted();
    assert_eq!(checked.kind(tail(&checked, &script)), &TyKind::Bool);
    let (checked, _) = Fixture::new().script("!1u8");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NotBool]);
}

/// operators §2: `&`, `<<` and `>>` are the language's own on the
/// integers alone; a shift's two operands may differ in width.
#[test]
fn bitwise_and_and_shifts_are_the_languages_own_on_the_integers() {
    let (checked, script) = Fixture::new().script("1u8 & 2");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
    let (checked, script) = Fixture::new().script("1u8 << 3u32");
    checked.admitted();
    assert!(is_int(&checked, tail(&checked, &script), IntTy::U8));
    let (checked, _) = Fixture::new().script("true & false");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NotInteger]);
    let (checked, _) = Fixture::new().script("1 >> true");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NotInteger]);
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
    let steps = checked.steps(value.id());
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].via.declaration, widen);
    assert!(is_int(&checked, steps[0].to, IntTy::I64));
    let TyKind::Fn { params, ret, .. } = checked.kind(steps[0].via_ty) else {
        panic!("a step's `via` is a function");
    };
    assert!(is_int(&checked, params[0], IntTy::U8));
    assert!(is_int(&checked, *ret, IntTy::I64));
    let tail_site = script.tail.as_ref().unwrap().id();
    assert!(!checked.typeck.coercions.contains_key(&tail_site));
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
        carries_effect: false,
        carries_identity: false,
        lifetimes: 0,
    });
    let other = fx.ty(TyKind::UserDefined {
        id: other_id,
        type_args: Box::new([]),
        effect: None,
        identity: None,
        marked: MarkTerm::Known(Mark::Unmarked),
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
    let Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. } = script.tail.as_deref().unwrap() else {
        panic!("not a call");
    };
    let steps = checked.steps(args[0].id());
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].via.declaration, as_slice);
    assert_eq!(steps[0].via.r#impl, Some(Impl::Concrete(1)));
    let TyKind::Fn { params, ret, .. } = checked.kind(steps[0].via_ty) else {
        panic!("a view's `via` is a function");
    };
    assert_eq!(params[0], args_ty(&checked, &args[0]));
    assert_eq!(*ret, steps[0].to);
}

/// control-flow, types §4, §6: a block whose last statement is a call of a
/// declaration whose result is `!` gives no value, which is known only once
/// its call is reached, and takes the type its use decides: the `if`'s.
#[test]
fn a_block_ending_in_a_call_of_a_declaration_of_result_never_takes_its_uses_type() {
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

/// functions §2: a capture names the binding it captured, by its binder's
/// site; here the outer `y`, not the shadowed one before it.
#[test]
fn a_capture_names_its_binder() {
    let fx = Fixture::new();
    let (checked, script) = fx.script("let y = 1; let y = true; let f = |z| -> y; f(1)");
    checked.admitted();
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    let binders: Vec<CaptureOf> = closure.captures.iter().map(|captured| captured.binder).collect();
    assert_eq!(binders, vec![CaptureOf::Binding(let_binder(&script, 1))]);
}

/// extern-fn §5-6: a reached call records its callee's function type as
/// instantiated at the call, at the site `reached` has.
#[test]
fn a_call_records_its_callees_type() {
    let mut fx = Fixture::new();
    let t = fx.var(0);
    let identity = fx.pure(&[t], t);
    fx.declare("id", identity, &[TyVarBound::Any]);
    let (checked, script) = fx.script("id(true)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    assert!(checked.typeck.reached.contains_key(&site));
    let TyKind::Fn { params, ret, .. } = checked.kind(checked.typeck.callees[&site]) else {
        panic!("a callee is a function type");
    };
    assert_eq!(checked.kind(params[0]), &TyKind::Bool);
    assert_eq!(checked.kind(*ret), &TyKind::Bool);
}

/// patterns §1: a name bound through a reference is a reference, and its
/// binder's type is the binding's, not the matched part's.
#[test]
fn a_binder_through_a_reference_has_the_bindings_type() {
    let fx = Fixture::new();
    let (checked, script) = fx.script("let p = &(1u8, true); let r = match p { (x, y) => x, }; 1");
    checked.admitted();
    let Stmt::LetBind { expr: Expr::Match { arms, .. }, .. } = &script.stmts[1] else {
        panic!("not a match");
    };
    let acvus_ast::Pattern::Tuple { elements, .. } = &arms[0].pattern else {
        panic!("not a tuple pattern");
    };
    let acvus_ast::TuplePatternElem::Pattern(x) = &elements[0] else {
        panic!("not a binding");
    };
    let TyKind::Ref(Mutability::Shared, target) = checked.kind(checked.typeck.types[&x.id()]) else {
        panic!("a binder through a `&` has a `&` type");
    };
    assert!(is_int(&checked, *target, IntTy::U8));
}

/// operators §3: an index is a `u64`, as Rust's `usize`.
/// An unsuffixed index takes it; an index of another width, which no cast
/// takes up to `u64`, has no conversion into it (types §3, §7).
#[test]
fn an_index_is_a_u64() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    let array = fx.ty(TyKind::Array(bool_ty, LenTerm::Known(2), ()));
    fx.context("xs", array);
    let (checked, script) = fx.script("@xs[0]");
    checked.admitted();
    let Some(Expr::Index { index, .. }) = script.tail.as_deref() else {
        panic!("not an index");
    };
    assert!(is_int(&checked, checked.typeck.types[&index.id()], IntTy::U64));
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    let array = fx.ty(TyKind::Array(bool_ty, LenTerm::Known(2), ()));
    fx.context("xs", array);
    let (checked, _) = fx.script("@xs[0u8]");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoConversion]);
}

/// context §3, effects §1: a context's load and commit are effects, so a
/// body that reads and writes a context and calls nothing has it among its
/// effect's reads and writes, opaque as an unknown load and commit are.
#[test]
fn a_bodys_effect_holds_the_contexts_it_touches() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.context("n", i64_ty);
    let n = QualifiedRef::root(fx.name("n"));
    let (checked, _) = fx.script("let v = @n; @n = v; 1");
    checked.admitted();
    assert!(checked.typeck.reached.is_empty());
    let effect = &checked.typeck.body_effect;
    assert_eq!(effect.reads(), &crate::ty::Contexts::from([n]));
    assert_eq!(effect.writes(), &crate::ty::Contexts::from([n]));
    assert_eq!(effect.reissue(), crate::ty::Reissue::Opaque);
}

/// extern-fn §6, operators §5: a view step's impl states its requirements,
/// and each is reached as a reached call's are, recorded on the step.
#[test]
fn a_view_steps_requirements_are_reached() {
    let mut fx = Fixture::new();
    let (vec_id, at_vec) = fx.vec_and_its_view();
    let (as_slice, clone) = (fx.raw.core.as_slice, fx.raw.core.clone);
    let e = fx.var(0);
    let shared_e = fx.reference(Mutability::Shared, e);
    // The pattern states the flows the impl it asks for has: an impl that
    // labels no lifetime takes every flow (extern-fn §4).
    let clones_e = fx.ty(TyKind::Fn {
        params: Box::new([shared_e]),
        ret: e,
        effect: EffectTerm::Known(crate::ty::Effect::pure()),
        flows: FlowTerm::Known(Flows::Every),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let view = Scheme {
        ty: at_vec,
        bounds: vec![TyVarBound::Any; variables_of(&fx.raw.types, &[at_vec])].into(),
        param_names: fx.param_names(1),
        impl_of: Some(ImplOf { generic_function: as_slice, kind: ImplKind::Concrete }),
        requires: Box::new([crate::decl::Requirement {
            generic_function: clone,
            pattern: clones_e,
            fact: None,
        }]),
        facts: Default::default(),
        runs: Box::new([]),
        awaits: false,
        specialized: Box::new([]),
    };
    fx.raw.functions.push((impl_id(&fx.names, as_slice, 0), view));
    let bool_ty = fx.ty(TyKind::Bool);
    let shared_bool = fx.reference(Mutability::Shared, bool_ty);
    let clones_bool = fx.pure(&[shared_bool], bool_ty);
    fx.lang_item(clone, &[clones_bool]);
    let bools = fx.ty(TyKind::Slice(bool_ty));
    let shared_bools = fx.reference(Mutability::Shared, bools);
    let unit = fx.ty(TyKind::Unit);
    let takes_slice = fx.pure(&[shared_bools], unit);
    fx.declare("first", takes_slice, &[]);
    let vec_bool = fx.vec_of(vec_id, bool_ty);
    fx.context("v", vec_bool);
    let (checked, script) = fx.script("first(&@v)");
    checked.admitted();
    let Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. } = script.tail.as_deref().unwrap() else {
        panic!("not a call");
    };
    let steps = checked.steps(args[0].id());
    assert_eq!(steps[0].via.r#impl, Some(Impl::Concrete(0)));
    let required: Vec<(QualifiedRef, Option<Impl>)> =
        steps[0].via.required.iter().map(|chosen| (chosen.declaration, chosen.r#impl)).collect();
    assert_eq!(required, vec![(clone, Some(Impl::Concrete(0)))]);
}

/// ownership §5, context §3: a pattern that binds into a context stores
/// into it, so the context's type has no positions, as for `@x = v`.
#[test]
fn a_pattern_never_binds_a_function_into_a_context() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let function = fx.pure(&[i64_ty], i64_ty);
    fx.context("f", function);
    let (checked, _) = fx.script("match @f { @f => 1, }");
    assert_eq!(checked.refusals(), vec![&RefusalKind::HoldsLoans]);
}

/// effects §1: a coercion is performed by its function, as a call is, so
/// that function's effect joins the body's, for a declared cast and for a
/// view alike.
#[test]
fn a_coercions_effect_joins_its_bodys() {
    use crate::ty::{Contexts, Effect, Reissue, SplitCost};
    let idempotent = || EffectTerm::Known(Effect::new(Reissue::Idempotent, true, Contexts::new(), Contexts::new(), SplitCost::Unavailable, false));

    let mut fx = Fixture::new();
    let (u8_ty, i64_ty) = (fx.int(IntTy::U8), fx.int(IntTy::I64));
    let widening = fx.function(&[u8_ty], i64_ty, idempotent());
    let widen = fx.declare("widen", widening, &[]);
    fx.raw.casts.push(CastRule {
        from: u8_ty,
        to: i64_ty,
        via: widen,
    });
    let (checked, _) = fx.script("if true { return 1u8; }; 2i64");
    checked.admitted();
    assert_eq!(checked.typeck.body_effect.reissue(), Reissue::Idempotent);

    let mut fx = Fixture::new();
    let (vec_id, _) = fx.vec_and_its_view();
    let e = fx.var(0);
    let vec_e = fx.vec_of(vec_id, e);
    let shared_vec_e = fx.reference(Mutability::Shared, vec_e);
    let slice_e = fx.ty(TyKind::Slice(e));
    let shared_slice_e = fx.reference(Mutability::Shared, slice_e);
    let view = fx.function(&[shared_vec_e], shared_slice_e, idempotent());
    let as_slice = fx.raw.core.as_slice;
    fx.lang_item(as_slice, &[view]);
    let bool_ty = fx.ty(TyKind::Bool);
    let bools = fx.ty(TyKind::Slice(bool_ty));
    let shared_bools = fx.reference(Mutability::Shared, bools);
    let unit = fx.ty(TyKind::Unit);
    let takes_slice = fx.pure(&[shared_bools], unit);
    fx.declare("first", takes_slice, &[]);
    let vec_bool = fx.vec_of(vec_id, bool_ty);
    let makes = fx.pure(&[], vec_bool);
    fx.declare("make", makes, &[]);
    let (checked, _) = fx.script("let v = make(); first(&v)");
    checked.admitted();
    assert_eq!(checked.typeck.body_effect.reissue(), Reissue::Idempotent);
}

/// operators §4: each part of a comparison names the function type its
/// `core::eq` impl has there, which lower calls.
#[test]
fn a_comparison_part_names_its_callee_type() {
    let mut fx = Fixture::new();
    let eq = fx.raw.core.eq;
    let bool_ty = fx.ty(TyKind::Bool);
    let i64_ty = fx.int(IntTy::I64);
    let shared = fx.reference(Mutability::Shared, i64_ty);
    let at_int = fx.pure(&[shared, shared], bool_ty);
    fx.lang_item(eq, &[at_int]);
    let (checked, script) = fx.script("(1, 2) == (3, 4)");
    checked.admitted();
    let site = script.tail.as_ref().unwrap().id();
    let super::StructureTree::Parts(parts) = &checked.typeck.structure_trees[&site] else {
        panic!("a tuple compares its parts");
    };
    for (_, part) in parts.iter() {
        let super::StructureTree::Reached { callee, .. } = part else {
            panic!("an integer reaches an impl");
        };
        let TyKind::Fn { params, ret, .. } = checked.kind(*callee) else {
            panic!("a callee is a function type");
        };
        let TyKind::Ref(Mutability::Shared, target) = checked.kind(params[0]) else {
            panic!("`eq` takes its operands by `&`");
        };
        assert!(is_int(&checked, *target, IntTy::I64));
        assert_eq!(checked.kind(*ret), &TyKind::Bool);
    }
}

/// functions §2, programs §4: an input read inside a lambda is captured,
/// as a binding is, by reference.
#[test]
fn an_input_read_in_a_lambda_is_captured() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    fx.input("n", i64_ty);
    let n = fx.name("n");
    let (checked, _) = fx.script("let f = |x| -> $n; f(1)");
    checked.admitted();
    let closure = checked.typeck.closures.values().next().expect("one lambda");
    assert_eq!(
        &closure.captures[..],
        &[Captured {
            name: n,
            binder: CaptureOf::Input(n),
            capture: Capture::Ref(Mutability::Shared)
        }]
    );
}

/// types §3: the run's result is the slot every tail and `return` enters,
/// and a template's is the `String` it builds.
#[test]
fn the_runs_result_type_is_recorded() {
    let (checked, _) = Fixture::new().script("if true { return 1u8; }; 2");
    checked.admitted();
    assert!(is_int(&checked, checked.typeck.result.expect("a script's result is known"), IntTy::U8));
    let (checked, _) = Fixture::new().template("% let x = 1");
    checked.admitted();
    assert_eq!(checked.kind(checked.typeck.result.expect("a template's result is known")), &TyKind::String);
}

/// D85: a captured name is typed as its binding inside the lambda, and
/// captured by how the lambda uses it: a `&mut` read only is captured `&`,
/// and one written through is captured `&mut`.
#[test]
fn a_mut_binding_used_read_only_in_a_lambda_is_captured_shared() {
    let capture_of = |source: &str| {
        let fx = Fixture::new();
        let (checked, script) = fx.script(source);
        checked.admitted();
        let Stmt::LetBind { expr, .. } = &script.stmts[2] else {
            panic!("not a let");
        };
        let closure = &checked.typeck.closures[&expr.id()];
        let [captured] = &closure.captures[..] else {
            panic!("one capture: {:?}", closure.captures);
        };
        (captured.capture, checked, script)
    };
    let (capture, checked, script) = capture_of("let x = 1; let r = &mut x; let f = |z| -> *r; f(0)");
    assert_eq!(capture, Capture::Ref(Mutability::Shared));
    let Stmt::LetBind { expr, .. } = &script.stmts[2] else {
        panic!("not a let");
    };
    let Expr::Lambda { body, .. } = expr else {
        panic!("not a lambda");
    };
    let Expr::UnaryOp { operand, .. } = &**body else {
        panic!("not a deref");
    };
    assert!(
        matches!(checked.kind(checked.typeck.types[&operand.id()]), TyKind::Ref(Mutability::Mut, _)),
        "the name inside is typed as its binding"
    );
    let (capture, _, _) = capture_of("let x = 1; let r = &mut x; let f = |z| -> { *r = 2; z }; f(0)");
    assert_eq!(capture, Capture::Ref(Mutability::Mut));
}

// Method receivers (types §6, D85)

/// An extern `Vec<T>` with both views, `len(&Vec<T>) -> i64`,
/// `push(&mut Vec<T>, T)` and `m(&mut String)`.
fn vec_with_methods(fx: &mut Fixture) -> (QualifiedRef, QualifiedRef, QualifiedRef, QualifiedRef) {
    let (vec_id, at_vec) = fx.vec_and_its_view();
    let as_slice = fx.raw.core.as_slice;
    fx.lang_item(as_slice, &[at_vec]);
    let e = fx.var(0);
    let vec_e = fx.vec_of(vec_id, e);
    let mut_vec_e = fx.reference(Mutability::Mut, vec_e);
    let slice_e = fx.ty(TyKind::Slice(e));
    let mut_slice_e = fx.reference(Mutability::Mut, slice_e);
    let at_vec_mut = fx.pure(&[mut_vec_e], mut_slice_e);
    let as_slice_mut = fx.raw.core.as_slice_mut;
    fx.lang_item(as_slice_mut, &[at_vec_mut]);
    let shared_vec_e = fx.reference(Mutability::Shared, vec_e);
    let i64_ty = fx.int(IntTy::I64);
    let unit = fx.ty(TyKind::Unit);
    let len_ty = fx.pure(&[shared_vec_e], i64_ty);
    let len = fx.declare("len", len_ty, &[TyVarBound::Any]);
    let push_ty = fx.pure(&[mut_vec_e, e], unit);
    let push = fx.declare("push", push_ty, &[TyVarBound::Any]);
    let string = fx.ty(TyKind::String);
    let mut_string = fx.reference(Mutability::Mut, string);
    let m_ty = fx.pure(&[mut_string], unit);
    let m = fx.declare("m", m_ty, &[]);
    (vec_id, len, push, m)
}

fn method_site(expr: &Expr) -> AstId {
    match expr {
        Expr::MethodCall { callee_id, .. } => *callee_id,
        other => panic!("not a method call: {other:?}"),
    }
}

/// types §6: `v.len()` reaches `len(&Vec)` through `&`, and `v.push(x)` on
/// a writable binding reaches `push(&mut Vec, T)` through `&mut`.
#[test]
fn a_receiver_is_lent_as_its_method_takes_it() {
    let mut fx = Fixture::new();
    let (vec_id, len, push, _) = vec_with_methods(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    let vec_i64 = fx.vec_of(vec_id, i64_ty);
    fx.context("v", vec_i64);
    let (checked, script) = fx.script("let v = @v; v.push(1); v.len()");
    checked.admitted();
    let Stmt::Expr(pushed) = &script.stmts[1] else {
        panic!("not an expression");
    };
    let pushed = method_site(pushed);
    assert_eq!(checked.typeck.reached[&pushed].declaration, push);
    assert_eq!(checked.typeck.adjustments[&pushed], Adjustment::Ref(Mutability::Mut));
    let measured = method_site(script.tail.as_deref().unwrap());
    assert_eq!(checked.typeck.reached[&measured].declaration, len);
    assert_eq!(checked.typeck.adjustments[&measured], Adjustment::Ref(Mutability::Shared));
}

/// types §6, D85: an input is no writable place, so its receiver is never
/// lent `&mut`, and `push` reaches nothing.
#[test]
fn an_input_receiver_is_never_lent_mut() {
    let mut fx = Fixture::new();
    let (vec_id, _, _, _) = vec_with_methods(&mut fx);
    let i64_ty = fx.int(IntTy::I64);
    let vec_i64 = fx.vec_of(vec_id, i64_ty);
    fx.input("v", vec_i64);
    let (checked, script) = fx.script("$v.push(1)");
    let site = method_site(script.tail.as_deref().unwrap());
    assert_ne!(checked.typeck.adjustments[&site], Adjustment::Ref(Mutability::Mut));
    assert!(matches!(checked.refusals()[..], [RefusalKind::NoDeclaration { .. }]), "{:?}", checked.refusals());
}

/// operators §3, D85: `v[i].m()` where `m` takes `&mut` reaches the
/// element through `as_slice_mut`, as Rust's `Index` becomes `IndexMut`.
#[test]
fn a_mut_receiver_indexes_through_as_slice_mut() {
    let mut fx = Fixture::new();
    let (vec_id, _, _, m) = vec_with_methods(&mut fx);
    let as_slice_mut = fx.raw.core.as_slice_mut;
    let string = fx.ty(TyKind::String);
    let vec_string = fx.vec_of(vec_id, string);
    fx.context("v", vec_string);
    let (checked, script) = fx.script("let v = @v; v[0].m()");
    checked.admitted();
    let Some(Expr::MethodCall { callee_id, receiver, .. }) = script.tail.as_deref() else {
        panic!("not a method call");
    };
    assert_eq!(checked.typeck.reached[callee_id].declaration, m);
    assert_eq!(checked.typeck.adjustments[callee_id], Adjustment::Ref(Mutability::Mut));
    let Expr::Index { callee_id: index, .. } = &**receiver else {
        panic!("not an index");
    };
    assert_eq!(checked.typeck.reached[index].declaration, as_slice_mut);
}

/// programs §1: a template's text is appended as written, and only a
/// `{{ }}` reaches `core::display`, so text needs no `display` impl.
#[test]
fn a_templates_text_reaches_no_display() {
    let mut fx = Fixture::new();
    fx.display(&[TyKind::Int(IntTy::I64)]);
    let (checked, _) = fx.template("count: {{ 42 }}\n");
    checked.admitted();
    assert_eq!(checked.typeck.reached.len(), 1);
}

/// programs §1: a `return` that would leave a template itself is refused,
/// since what it leaves with is not the `String` the template builds; one
/// inside a lambda leaves the lambda.
#[test]
fn a_return_in_a_template_is_refused() {
    let (checked, _) = Fixture::new().template("% return 1\n");
    assert_eq!(checked.refusals(), vec![&RefusalKind::ReturnInTemplate]);
    let (checked, _) = Fixture::new().template("% let f = |x| -> return x\n");
    checked.admitted();
}

/// programs §1: a `?` that would leave a template itself is refused as a
/// `return` is.
#[test]
fn a_try_leaving_a_template_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let option = fx.ty(TyKind::Option(i64_ty, ()));
    fx.context("o", option);
    let (checked, _) = fx.template("% let x = @o?\n");
    assert_eq!(checked.refusals(), vec![&RefusalKind::ReturnInTemplate]);
}

/// types §5: a function type's flows nothing decides state every flow. The
/// default is taken at a stall, with effects and marks, so the solved terms
/// hold it before anything is frozen; a lambda's own flow set is its claim's
/// and is still open there.
#[test]
fn an_unclaimed_flow_set_takes_every_flow_at_a_stall() {
    let fx = Fixture::new();
    let script = acvus_ast::parse_script(&fx.names, "let f = |g| -> g(1); 1").expect("the test's source parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let interfaces = super::Interfaces::new();
    let mut cx = super::ctxt::Ctxt::new(&fx.names, &decls, &interfaces, super::scope::Run::Script);
    super::stmt::check_body(&mut cx, &script.stmts, script.tail.as_deref());
    super::decision::solve(&mut cx);
    assert!(cx.out.refusals.is_empty(), "refused: {:?}", cx.out.refusals);

    let lambda = *cx.out.closures.keys().next().expect("the script writes one lambda");
    let freeze = |cx: &mut super::ctxt::Ctxt<'_>, term| {
        let mut types = Types::new();
        let frozen = cx.infer.freeze(&mut types, &mut crate::infer::FreezeMemo::default(), term);
        frozen.map(|ty| types.kind(ty).clone())
    };
    let lambda_term = cx.out.types[&lambda];
    assert!(matches!(freeze(&mut cx, lambda_term), Err(crate::infer::Unsettled::Flow(_))));
    let values: Vec<_> = cx.out.types.iter().filter(|(site, _)| **site != lambda).map(|(_, term)| *term).collect();
    let called = values
        .into_iter()
        .filter_map(|term| match freeze(&mut cx, term) {
            Ok(TyKind::Fn { flows, .. }) => Some(flows),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!called.is_empty(), "`g` is a function value");
    assert!(called.iter().all(|flows| *flows == FlowTerm::Known(Flows::Every)), "{called:?}");
}



// Round two of the structure rule (types §1, §3, §6; operators §2, §4;
// ownership §2, §5)

/// A generic function `name` whose one concrete impl is `at`, both of the
/// signature `generic`.
fn generic_with_impl(fx: &mut Fixture, name: &str, generic: TyId<Poly>, at: TyId<Poly>, facts: DeclaredFacts) -> QualifiedRef {
    let id = QualifiedRef::root(fx.name(name));
    let arity = fx.arity(generic);
    let at_impl = Scheme {
        ty: at,
        bounds: vec![TyVarBound::Any; variables_of(&fx.raw.types, &[at])].into(),
        param_names: fx.param_names(arity),
        impl_of: Some(ImplOf { generic_function: id, kind: ImplKind::Concrete }),
        requires: Box::new([]),
        facts,
        runs: Box::new([]),
        specialized: Box::new([]),
        awaits: false,
    };
    let function = GenericFunction { ty: generic, param_names: fx.param_names(arity), bounds: Box::new([TyVarBound::Any]), specialized: Box::new([]) };
    fx.raw.functions.push((impl_id(&fx.names, id, 0), at_impl));
    fx.raw.generic_functions.push((id, function));
    id
}

/// types §1, §6: a value of a structured type a declaration wrote binds
/// the slot of an `if` first, whichever branch it is in, and the other
/// branch's value is coerced to it: the outcome does not depend on the
/// order the branches are checked in.
#[test]
fn a_meeting_is_decided_by_the_value_written_elsewhere_in_either_order() {
    let tuple_fixture = || {
        let mut fx = Fixture::new();
        let (u8_ty, i64_ty, bool_ty) = (fx.int(IntTy::U8), fx.int(IntTy::I64), fx.ty(TyKind::Bool));
        let widening = fx.pure(&[u8_ty], i64_ty);
        let widen = fx.declare("widen", widening, &[]);
        fx.raw.casts.push(CastRule { from: u8_ty, to: i64_ty, via: widen });
        let pair = fx.ty(TyKind::Tuple(Box::new([i64_ty, bool_ty]), ()));
        let makes = fx.pure(&[], pair);
        fx.declare("f", makes, &[]);
        fx.input("c", bool_ty);
        fx
    };
    for source in ["if $c { f() } else { (1u8, true) }", "if $c { (1u8, true) } else { f() }"] {
        let (checked, _) = tuple_fixture().script(source);
        checked.admitted();
    }
    let object_fixture = || {
        let mut fx = Fixture::new();
        let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
        let a = fx.name("a");
        let object = fx.ty(TyKind::Object { fields: Box::new([(a, i64_ty)]), label: () });
        let makes = fx.pure(&[], object);
        fx.declare("f", makes, &[]);
        fx.input("c", bool_ty);
        fx
    };
    for source in ["if $c { f() } else { { a: 1, b: true, } }", "if $c { { a: 1, b: true, } } else { f() }"] {
        let (checked, _) = object_fixture().script(source);
        checked.admitted();
    }
}

/// types §1: at an enum, `Option` and `Result`, a rebuilt value's variant
/// is its one argument's; a function taking the type in two parameters
/// has no lifted impl there, and does at a tuple.
#[test]
fn a_rebuild_lifts_at_a_sum_only_from_one_argument() {
    for (source, admitted) in [("pick(&(1, 2), &(3, 4))", true), ("pick(&Some(1), &Some(2))", false)] {
        let mut fx = Fixture::new();
        let (t, i64_ty) = (fx.var(0), fx.int(IntTy::I64));
        let (shared_t, shared_int) = (fx.reference(Mutability::Shared, t), fx.reference(Mutability::Shared, i64_ty));
        let generic = fx.pure(&[shared_t, shared_t], t);
        let at_int = fx.pure(&[shared_int, shared_int], i64_ty);
        let pick = generic_with_impl(&mut fx, "pick", generic, at_int, DeclaredFacts::default());
        let (checked, _) = fx.script(source);
        match admitted {
            true => checked.admitted(),
            false => assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: pick.name }]),
        }
    }
}

/// operators §2: `Option`'s `default` is `None`, the language's own value;
/// a structure of `Option` parts defaults each so. `Result` and a
/// program's enum have none, nor has a structure with a `Result` part.
#[test]
fn default_is_none_at_an_option_and_absent_at_a_result_or_an_enum() {
    let check = |param: &dyn Fn(&mut Fixture, TyId<Poly>) -> TyId<Poly>| {
        let mut fx = Fixture::new();
        let i64_ty = fx.int(IntTy::I64);
        let param = param(&mut fx, i64_ty);
        let unit = fx.ty(TyKind::Unit);
        let takes = fx.pure(&[param], unit);
        fx.declare("f", takes, &[]);
        fx.script("f(default())").0
    };
    let option = check(&|fx, int| fx.ty(TyKind::Option(int, ())));
    option.admitted();
    let pair = check(&|fx, int| {
        let option = fx.ty(TyKind::Option(int, ()));
        fx.ty(TyKind::Tuple(Box::new([option, option]), ()))
    });
    pair.admitted();
    let refused = |checked: &Checked| matches!(checked.refusals()[..], [RefusalKind::NoDeclaration { .. }]);
    let result = check(&|fx, int| fx.ty(TyKind::Result(int, int, ())));
    assert!(refused(&result), "{:?}", result.refusals());
    let with_result = check(&|fx, int| {
        let option = fx.ty(TyKind::Option(int, ()));
        let result = fx.ty(TyKind::Result(int, int, ()));
        fx.ty(TyKind::Tuple(Box::new([option, result]), ()))
    });
    assert!(refused(&with_result), "{:?}", with_result.refusals());
    let color = check(&|fx, _| {
        let red = fx.name("Red");
        fx.ty(TyKind::Enum { name: QualifiedRef::root(fx.name("Color")), variants: Box::new([(red, None)]), label: () })
    });
    assert!(refused(&color), "{:?}", color.refusals());
}

/// operators §4: an enum, `Option` and `Result` have equality but no
/// order, so ordering one is refused, as a part too.
#[test]
fn ordering_a_sum_is_refused() {
    for (source, parts) in [("Some(1) < Some(2)", 1), ("(Some(1), Ok(2)) >= (Some(1), Ok(2))", 2), ("let c = if true { Color::Red } else { Color::Blue }; c < c", 1)] {
        let (checked, _) = Fixture::new().script(source);
        assert_eq!(checked.refusals(), vec![&RefusalKind::Unordered; parts], "{source}");
    }
    let (checked, _) = Fixture::new().script("Some(true) == Some(false)");
    assert!(!checked.refusals().contains(&&RefusalKind::Unordered));
}

/// ownership §2: a value that copies clones by its copy, with no `clone`
/// impl, alone or as a part; a reference part copies.
#[test]
fn a_value_that_copies_clones_by_its_copy() {
    for source in ["1.clone()", "let x = 1; let y = (&x, 1).clone(); 0"] {
        let (checked, _) = Fixture::new().script(source);
        checked.admitted();
        let copied: Vec<_> = checked.typeck.reached.iter().filter(|(_, chosen)| chosen.r#impl == Some(Impl::Structural)).collect();
        let [(site, _)] = copied[..] else {
            panic!("one call is answered by the language: {copied:?}");
        };
        assert_eq!(checked.typeck.structure_trees[site], super::StructureTree::Own(super::Own::Copy), "{source}");
    }
}

/// Unimplemented, not a rule: a requirement a lift would answer is
/// refused, since no declared function stands for the lifted impl.
#[test]
fn a_requirement_a_lift_would_answer_is_refused_as_unimplemented() {
    let mut fx = Fixture::new();
    let (vec_id, at_vec) = fx.vec_and_its_view();
    let (as_slice, clone) = (fx.raw.core.as_slice, fx.raw.core.clone);
    let e = fx.var(0);
    let shared_e = fx.reference(Mutability::Shared, e);
    let clones_e = fx.ty(TyKind::Fn {
        params: Box::new([shared_e]),
        ret: e,
        effect: EffectTerm::Known(crate::ty::Effect::pure()),
        flows: FlowTerm::Known(Flows::Every),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let view = Scheme {
        ty: at_vec,
        bounds: vec![TyVarBound::Any; variables_of(&fx.raw.types, &[at_vec])].into(),
        param_names: fx.param_names(1),
        impl_of: Some(ImplOf { generic_function: as_slice, kind: ImplKind::Concrete }),
        requires: Box::new([crate::decl::Requirement { generic_function: clone, pattern: clones_e, fact: None }]),
        facts: Default::default(),
        runs: Box::new([]),
        awaits: false,
        specialized: Box::new([]),
    };
    fx.raw.functions.push((impl_id(&fx.names, as_slice, 0), view));
    let string = fx.ty(TyKind::String);
    fx.clones(&[string]);
    let single = fx.ty(TyKind::Tuple(Box::new([string]), ()));
    let singles = fx.ty(TyKind::Slice(single));
    let shared_singles = fx.reference(Mutability::Shared, singles);
    let unit = fx.ty(TyKind::Unit);
    let takes_slice = fx.pure(&[shared_singles], unit);
    fx.declare("first", takes_slice, &[]);
    let vec_single = fx.vec_of(vec_id, single);
    fx.context("v", vec_single);
    let (checked, _) = fx.script("first(&@v)");
    let feature = super::Unimplemented::LiftedImplAsRequirement;
    assert_eq!(checked.refusals(), vec![&RefusalKind::Unimplemented { feature }]);
}

/// extern-fn §3, §6, types §1: `cmp` combines its parts' answers
/// lexicographically, which keeps a total order, so a tuple has a lifted
/// `cmp` only where every part's `cmp` is declared one.
#[test]
fn cmp_lifts_only_where_every_parts_cmp_is_declared_a_total_order() {
    for states in [false, true] {
        let mut fx = Fixture::new();
        let cmp = fx.raw.core.cmp;
        let i64_ty = fx.int(IntTy::I64);
        let shared = fx.reference(Mutability::Shared, i64_ty);
        let at_int = fx.pure(&[shared, shared], i64_ty);
        let action: Box<[crate::decl::declared::Action]> = match states {
            true => Box::new([crate::decl::declared::Action::TotalOrder]),
            false => Box::new([]),
        };
        fx.lang_item_stating(cmp, &[at_int], DeclaredFacts { action, ..Default::default() });
        let pair = fx.ty(TyKind::Tuple(Box::new([i64_ty, i64_ty]), ()));
        fx.input("t", pair);
        let (checked, _) = fx.script("cmp(&$t, &$t)");
        match states {
            true => checked.admitted(),
            false => assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: cmp.name }; 2]),
        }
    }
}

/// D82: a law is stated by the declared fact of its kind, and no longer by
/// the old `laws` form.
#[test]
fn a_law_is_stated_by_its_declared_fact() {
    use crate::decl::NamedFact;
    use crate::decl::declared::{Access, Action};
    let equivalence = DeclaredFacts { access: Box::new([Access::Equivalence]), ..Default::default() };
    let total_order = DeclaredFacts { action: Box::new([Action::TotalOrder]), ..Default::default() };
    assert!(NamedFact::Equivalence.stated_by(&equivalence));
    assert!(!NamedFact::Equivalence.stated_by(&total_order));
    assert!(NamedFact::TotalOrder.stated_by(&total_order));
    assert!(!NamedFact::TotalOrder.stated_by(&equivalence));
}

/// ownership §5, types §3: a reference to a structured value entering a
/// parameter that is a reference to a declaration's structure is coerced
/// through its target, each part as a reference into the original
/// storage: an object narrows, but a part is not cast, since a reference
/// to a `u8` is no reference to an `i64`; through a `&mut`, an enum does
/// not widen.
#[test]
fn a_reference_to_a_structure_is_coerced_through_its_target() {
    let mut fx = Fixture::new();
    let (i64_ty, unit) = (fx.int(IntTy::I64), fx.ty(TyKind::Unit));
    let a = fx.name("a");
    let object = fx.ty(TyKind::Object { fields: Box::new([(a, i64_ty)]), label: () });
    let shared_object = fx.reference(Mutability::Shared, object);
    let takes = fx.pure(&[shared_object], unit);
    fx.declare("f", takes, &[]);
    let (checked, script) = fx.script("f(&{ a: 1, b: true, })");
    checked.admitted();
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("not a call");
    };
    let super::Coercion::Structured { through, shape, parts, .. } = &checked.typeck.coercions[&args[0].id()] else {
        panic!("a reference to an object is coerced through its target");
    };
    assert_eq!(*through, Some(Mutability::Shared));
    assert!(matches!(shape, ShapeCoercion::Narrowed { .. }), "{shape:?}");
    assert!(matches!(parts[..], [(PartKey::Field(_), None)]), "{parts:?}");

    let mut fx = Fixture::new();
    let (u8_ty, i64_ty, bool_ty, unit) = (fx.int(IntTy::U8), fx.int(IntTy::I64), fx.ty(TyKind::Bool), fx.ty(TyKind::Unit));
    let widening = fx.pure(&[u8_ty], i64_ty);
    let widen = fx.declare("widen", widening, &[]);
    fx.raw.casts.push(CastRule { from: u8_ty, to: i64_ty, via: widen });
    let pair = fx.ty(TyKind::Tuple(Box::new([i64_ty, bool_ty]), ()));
    let shared_pair = fx.reference(Mutability::Shared, pair);
    let takes = fx.pure(&[shared_pair], unit);
    fx.declare("f", takes, &[]);
    let (checked, _) = fx.script("f(&(1u8, true))");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoConversion]);

    for (mutability, source, admitted) in [
        (Mutability::Shared, "let c = Color::Red; g(&c)", true),
        (Mutability::Mut, "let c = Color::Red; g(&mut c)", false),
    ] {
        let mut fx = Fixture::new();
        let (red, blue, unit) = (fx.name("Red"), fx.name("Blue"), fx.ty(TyKind::Unit));
        let mut variants = [(blue, None), (red, None)];
        variants.sort_by_key(|(tag, _)| *tag);
        let color = fx.ty(TyKind::Enum { name: QualifiedRef::root(fx.name("Color")), variants: Box::new(variants), label: () });
        let lent = fx.reference(mutability, color);
        let takes = fx.pure(&[lent], unit);
        fx.declare("g", takes, &[]);
        let (checked, _) = fx.script(source);
        match admitted {
            true => checked.admitted(),
            false => assert!(matches!(checked.refusals()[..], [RefusalKind::Mismatch(FrozenMismatch { mismatch: Mismatch::Heads { .. }, .. })]), "{:?}", checked.refusals()),
        }
    }
}

/// types §6: where the arguments leave more than one, the type the result
/// is taken at decides last, exactly: `default()` where a `bool` is needed
/// reaches the `bool` impl, and where an `(i64, bool)` is needed, the
/// lifted impl, a candidate among the concrete ones (types §1).
#[test]
fn the_result_decides_a_reach_last() {
    let fixture = |param: &dyn Fn(&mut Fixture) -> TyId<Poly>| {
        let mut fx = Fixture::new();
        let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
        let default = fx.raw.core.default;
        let (at_int, at_bool) = (fx.pure(&[], i64_ty), fx.pure(&[], bool_ty));
        fx.lang_item(default, &[at_int, at_bool]);
        let param = param(&mut fx);
        let unit = fx.ty(TyKind::Unit);
        let takes = fx.pure(&[param], unit);
        fx.declare("f", takes, &[]);
        fx
    };
    let (checked, script) = fixture(&|fx| fx.ty(TyKind::Bool)).script("f(default())");
    checked.admitted();
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("not a call");
    };
    let Expr::FuncCall { id, .. } = &args[0] else {
        panic!("`default()` is a call");
    };
    assert_eq!(checked.typeck.reached[id].r#impl, Some(Impl::Concrete(1)));

    let pair = |fx: &mut Fixture| {
        let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
        fx.ty(TyKind::Tuple(Box::new([i64_ty, bool_ty]), ()))
    };
    let (checked, script) = fixture(&pair).script("f(default())");
    checked.admitted();
    let Some(Expr::FuncCall { args: acvus_ast::Args::Positional(args), .. }) = script.tail.as_deref() else {
        panic!("not a call");
    };
    let Expr::FuncCall { id, .. } = &args[0] else {
        panic!("`default()` is a call");
    };
    assert_eq!(checked.typeck.reached[id].r#impl, Some(Impl::Structural));
}

/// types §6: where the result would decide and its type never becomes
/// known, the reach is refused as never known.
#[test]
fn a_reach_its_result_would_decide_and_never_does_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let default = fx.raw.core.default;
    let at_int = fx.pure(&[], i64_ty);
    fx.lang_item(default, &[at_int]);
    let (checked, _) = fx.script("let x = default(); 0");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NeverKnown { open: default.name }]);
}

/// types §1, §6: a lift is reached only by an argument taken as the lift
/// takes its variable: `clone`'s `&T` by a `&`, and neither by a `&mut` nor
/// by the value; a method call's receiver is adjusted to it by `&`.
#[test]
fn a_lift_is_reached_only_as_it_takes_its_variable() {
    for (source, admitted) in [("clone(&$t)", true), ("clone(&mut $t)", false), ("clone($t)", false), ("$t.clone()", true)] {
        let mut fx = Fixture::new();
        let clone = fx.raw.core.clone;
        let string = fx.ty(TyKind::String);
        fx.clones(&[string]);
        let single = fx.ty(TyKind::Tuple(Box::new([string, string]), ()));
        fx.input("t", single);
        let (checked, _) = fx.script(source);
        match admitted {
            true => checked.admitted(),
            false => assert!(
                checked.refusals().contains(&&RefusalKind::NoDeclaration { name: clone.name }),
                "{source}: {:?}",
                checked.refusals()
            ),
        }
    }
    for source in ["(1, 2).clone()", "1.clone()"] {
        let (checked, _) = Fixture::new().script(source);
        checked.admitted();
    }
}

// G, let-bound function values (functions §2)

/// functions §2: a function value bound by `let` is polymorphic; each use
/// instantiates it.
#[test]
fn a_let_bound_lambda_is_polymorphic() {
    let (checked, _) = Fixture::new().script("let f = |x| -> x; let a = f(1); let b = f(true); 0");
    checked.admitted();
    assert_eq!(checked.typeck.instances.len(), 2);
}

/// Declares `a` over `i64` and `u8`, and `b` over `i64` and `bool`, each
/// returning an `i64`: overloads, so a call of either waits for its
/// argument's type.
fn methods_a_and_b(fx: &mut Fixture) {
    let i64_ty = fx.int(IntTy::I64);
    for (name, widths) in [("a", [TyKind::Int(IntTy::I64), TyKind::Int(IntTy::U8)]), ("b", [TyKind::Int(IntTy::I64), TyKind::Bool])] {
        for (at, width) in widths.into_iter().enumerate() {
            let param = fx.ty(width);
            let ty = fx.pure(&[param], i64_ty);
            fx.overload(&format!("n{at}"), name, ty);
        }
    }
}

/// functions §2: two function values joined into one slot require what
/// either requires, so a use at a type only one answers is refused.
#[test]
fn a_join_of_function_values_requires_what_either_requires() {
    let mut fx = Fixture::new();
    methods_a_and_b(&mut fx);
    let b = fx.name("b");
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let source = "let f = |x| -> a(x); let k = |x| -> b(x); let h = if $c { f } else { k }; let p = h(1i64); let q = h(1u8); 0";
    let (checked, _) = fx.script(source);
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: b }]);
}

/// functions §2: a binding's scheme is settled from every value it is
/// given, so a use written before an assignment requires what the
/// assigned value does too.
#[test]
fn a_bindings_scheme_is_settled_from_every_value_it_is_given() {
    let mut fx = Fixture::new();
    let (i64_ty, string, bool_ty, unit) = (fx.int(IntTy::I64), fx.ty(TyKind::String), fx.ty(TyKind::Bool), fx.ty(TyKind::Unit));
    for (namespace, target) in [("s", string), ("b", bool_ty)] {
        let lent = fx.reference(Mutability::Mut, target);
        let pushes = fx.pure(&[lent, i64_ty], unit);
        fx.overload(namespace, "push", pushes);
    }
    let lent_string = fx.reference(Mutability::Mut, string);
    let clears = fx.pure(&[lent_string], unit);
    let clear = fx.overload("s", "clear", clears);
    fx.context("t", string);
    fx.context("u", bool_ty);
    let source = "let step = |s| -> s.push(1); let x = step(&mut @u); step = |s| -> s.clear(); let y = step(&mut @t); 0";
    let (checked, _) = fx.script(source);
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: clear.name }]);
}

/// functions §2: a value whose body is elsewhere, a call's result, is not
/// generalized: its instance was fixed where it was made.
#[test]
fn a_calls_result_is_not_generalized() {
    let mut fx = Fixture::new();
    let (i64_ty, f64_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Float));
    let adds = fx.pure(&[i64_ty], i64_ty);
    let makes = fx.pure(&[i64_ty], adds);
    fx.declare("make_adder", makes, &[]);
    let _ = f64_ty;
    let (checked, _) = fx.script("let m = make_adder(1); let a = m(1); let b = m(true); 0");
    assert!(!checked.refusals().is_empty());
    assert!(checked.typeck.instances.is_empty());
}

/// functions §2, types §1: what a captured binding holds belongs to where
/// it was written, so a lambda that stores its parameter into one is not
/// generalized over it.
#[test]
fn a_captured_bindings_type_is_not_generalized() {
    let (checked, _) = Fixture::new().script("let z; let f = |x| -> { z = x; x }; let a = f(1); let b = f(true); 0");
    assert!(!checked.refusals().is_empty(), "`z` is one binding of one type");
}

/// functions §2: a requirement refused at a use names the site it arose
/// at, in the lambda's body.
#[test]
fn a_requirement_refused_at_a_use_names_where_it_arose() {
    let mut fx = Fixture::new();
    methods_a_and_b(&mut fx);
    let a = fx.name("a");
    let source = "let f = |x| -> a(x); let p = f(1u8); let q = f(true); 0";
    let (checked, _) = fx.script(source);
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoDeclaration { name: a }]);
    let at = source.find("a(x)").expect("the call is written");
    assert_eq!(checked.typeck.refusals[0].span.start, at);
}


/// functions §2: a use of a generalized binding inside another's value is
/// an instance within each instance of the other.
#[test]
fn a_use_inside_a_generalized_value_is_an_instance_within_its_instances() {
    let (checked, _) = Fixture::new().script("let f = |x| -> x; let g = |y| -> f(y); let a = g(1); let b = g(true); 0");
    checked.admitted();
    let within = checked.typeck.instances.values().filter(|instance| instance.within.is_some()).count();
    assert_eq!(within, 2, "`f` within each of `g`'s two instances");
}

/// functions §2: a length the type leaves unconstrained is generalized as
/// a type is, so one lambda takes arrays of two lengths.
#[test]
fn a_length_is_generalized() {
    for source in [
        "let f = |a| -> a; let p = f([1, 2]); let q = f([1, 2, 3]); 0",
        "let f = |a| -> match a { [x, ..] => x, }; let p = f([1, 2]); let q = f([1, 2, 3]); 0",
    ] {
        let (checked, _) = Fixture::new().script(source);
        checked.admitted();
        assert_eq!(checked.typeck.instances.len(), 2, "{source}");
    }
}

/// functions §2, effects §1: an effect the type leaves unconstrained is
/// generalized, its flows copied, so each instance's effect is what its
/// own calls do: effectful at one, pure at the other.
#[test]
fn each_instance_has_its_own_effect() {
    let mut fx = Fixture::new();
    let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    let quiet = fx.pure(&[i64_ty], i64_ty);
    let loud = fx.function(&[bool_ty], i64_ty, EffectTerm::Known(crate::ty::Effect::opaque()));
    fx.overload("q", "h", quiet);
    fx.overload("l", "h", loud);
    let (checked, script) = fx.script("let g = |x| -> h(x); let p = g(1); let q = g(true); 0");
    checked.admitted();
    let Stmt::LetBind { expr, .. } = &script.stmts[0] else {
        panic!("not a let");
    };
    let mut reissues: Vec<crate::ty::Reissue> = checked
        .typeck
        .instances
        .values()
        .map(|instance| match checked.kind(instance.types[&expr.id()]) {
            TyKind::Fn { effect: EffectTerm::Known(effect), .. } => effect.reissue(),
            other => panic!("the lambda is a function at each instance, not {other:?}"),
        })
        .collect();
    reissues.sort_by_key(|reissue| *reissue as u8);
    assert_eq!(reissues, vec![crate::ty::Reissue::Pure, crate::ty::Reissue::Opaque]);
}

/// The flows a function type of `kind` states.
fn flows_of(kind: &TyKind<Concrete>) -> crate::ty::Flows {
    match kind {
        TyKind::Fn { flows: FlowTerm::Known(flows), .. } => flows.clone(),
        other => panic!("a function type of known flows, not {other:?}"),
    }
}

/// The one flow of a function returning what its first parameter holds.
fn result_from_first_param() -> crate::ty::Flows {
    crate::ty::Flows::of([crate::ty::Flow { to: crate::ty::FlowEnd::Result, from: crate::ty::FlowEnd::Param(0), alignment: crate::ty::Alignment::Any }])
}

/// functions §2, ownership §5: a function type's flows are generalized, so
/// each instance of a lambda claims its own at its own types: at a
/// reference its result takes from its parameter, at a value it has no
/// flow.
#[test]
fn each_instance_of_a_lambda_claims_its_own_flows() {
    let (checked, script) = Fixture::new().script("let id = |x| -> x; let a = 1; let p = id(&a); let q = id(1); 0");
    checked.admitted();
    let Stmt::LetBind { expr, .. } = &script.stmts[0] else {
        panic!("not a let");
    };
    let mut claimed: Vec<(bool, crate::ty::Flows)> = checked
        .typeck
        .instances
        .values()
        .map(|instance| {
            let signature = checked.kind(instance.types[&expr.id()]);
            let TyKind::Fn { params, .. } = signature else {
                panic!("the lambda is a function at each instance, not {signature:?}");
            };
            (matches!(checked.kind(params[0]), TyKind::Ref(..)), flows_of(signature))
        })
        .collect();
    claimed.sort_by_key(|(reference, _)| *reference);
    assert_eq!(claimed, vec![(false, crate::ty::Flows::none()), (true, result_from_first_param())]);
}

/// graph, ownership §5: a unit is generalized as a let-bound function
/// value is, its flows among what it generalizes, so each call claims its
/// own at its own types. A unit's result holds no loan (a run's result),
/// so its only output is its `&mut` parameter, which the claim, flow
/// insensitive, has take from every other input holding a position: from
/// the other parameter at a reference, from nothing at a value.
#[test]
fn each_call_of_a_unit_claims_its_own_flows() {
    let source = "let a = 1; let c = 2; let p = lib::put(m: &mut a, r: &c); let d = 3; let q = lib::put(m: &mut d, r: 4); 0";
    let (checked, script, _) = Fixture::new().script_with_units(&[(&["lib"], "put", "{ let x = $r; *$m = 0; 0 }")], source);
    checked.admitted();
    let callee = |at: usize| {
        let Stmt::LetBind { expr, .. } = &script.stmts[at] else {
            panic!("not a let");
        };
        checked.kind(checked.typeck.callees[&expr.id()]).clone()
    };
    let into_m_from_r = crate::ty::Flows::of([crate::ty::Flow {
        to: crate::ty::FlowEnd::Param(0),
        from: crate::ty::FlowEnd::Param(1),
        alignment: crate::ty::Alignment::Any,
    }]);
    assert_eq!(flows_of(&callee(2)), into_m_from_r);
    assert_eq!(flows_of(&callee(4)), crate::ty::Flows::none());
}

/// functions §2: one rule for every use, so a binding used once is one
/// instance.
#[test]
fn a_binding_used_once_is_one_instance() {
    let (checked, _) = Fixture::new().script("let f = |x| -> x; f(1)");
    checked.admitted();
    assert_eq!(checked.typeck.instances.len(), 1);
}

// as_ref and as_mut (operators §2)

/// operators §2: `as_ref` lends each part of a transparent structured
/// value `&`, one level: the receiver is taken by `&`, and the result is
/// the structure rebuilt of references, an inner structure a `&` itself.
#[test]
fn as_ref_lends_each_part_one_level() {
    let cases: [(&str, &dyn Fn(&Checked, Ty) -> bool); 7] = [
        ("let p = (1, true); let r = p.as_ref(); 0", &|checked, ty| {
            matches!(checked.kind(ty), TyKind::Tuple(items, ()) if items.len() == 2 && items.iter().all(|item| matches!(checked.kind(*item), TyKind::Ref(Mutability::Shared, _))))
        }),
        ("let p = { a: 1, b: true, }; let r = p.as_ref(); 0", &|checked, ty| {
            matches!(checked.kind(ty), TyKind::Object { fields, label: () } if fields.iter().all(|(_, field)| matches!(checked.kind(*field), TyKind::Ref(Mutability::Shared, _))))
        }),
        ("let o = Some(1); let r = o.as_ref(); 0", &|checked, ty| {
            matches!(checked.kind(ty), TyKind::Option(payload, ()) if matches!(checked.kind(*payload), TyKind::Ref(Mutability::Shared, _)))
        }),
        ("let o = if true { Ok(1) } else { Err(true) }; let r = o.as_ref(); 0", &|checked, ty| {
            matches!(checked.kind(ty), TyKind::Result(ok, err, ()) if [ok, err].iter().all(|part| matches!(checked.kind(**part), TyKind::Ref(Mutability::Shared, _))))
        }),
        ("let c = Color::Red(1); let r = c.as_ref(); 0", &|checked, ty| {
            matches!(checked.kind(ty), TyKind::Enum { variants, .. } if variants.iter().all(|(_, payload)| payload.is_some_and(|payload| matches!(checked.kind(payload), TyKind::Ref(Mutability::Shared, _)))))
        }),
        ("let a = [1, 2]; let r = a.as_ref(); 0", &|checked, ty| {
            matches!(checked.kind(ty), TyKind::Array(element, len, ()) if crate::ir::known_len(len) == 2 && matches!(checked.kind(*element), TyKind::Ref(Mutability::Shared, _)))
        }),
        ("let p = ((1, true), 2); let r = p.as_ref(); 0", &|checked, ty| {
            let TyKind::Tuple(items, ()) = checked.kind(ty) else { return false };
            matches!(checked.kind(items[0]), TyKind::Ref(Mutability::Shared, inner) if matches!(checked.kind(*inner), TyKind::Tuple(..)))
        }),
    ];
    for (source, fits) in cases {
        let fx = Fixture::new();
        let names = fx.names.clone();
        let (checked, script) = fx.script(source);
        checked.admitted();
        assert!(fits(&checked, binding(&checked, &script, &names, "r")), "{source}: {:?}", checked.kind(binding(&checked, &script, &names, "r")));
    }
}

/// operators §2, types §4: `as_mut` lends each part `&mut`; under
/// `as_ref`, a `&mut` part is read shared.
#[test]
fn as_mut_lends_mut_and_a_mut_part_under_as_ref_reads_shared() {
    let fx = Fixture::new();
    let names = fx.names.clone();
    let (checked, script) = fx.script("let p = (1, true); let r = p.as_mut(); 0");
    checked.admitted();
    let TyKind::Tuple(items, ()) = checked.kind(binding(&checked, &script, &names, "r")) else {
        panic!("a tuple of references");
    };
    assert!(items.iter().all(|item| matches!(checked.kind(*item), TyKind::Ref(Mutability::Mut, _))));

    let fx = Fixture::new();
    let names = fx.names.clone();
    let (checked, script) = fx.script("let a = 1; let p = (&mut a, 2); let r = p.as_ref(); 0");
    checked.admitted();
    let TyKind::Tuple(items, ()) = checked.kind(binding(&checked, &script, &names, "r")) else {
        panic!("a tuple of references");
    };
    let TyKind::Ref(Mutability::Shared, target) = checked.kind(items[0]) else {
        panic!("the `&mut` part is reborrowed shared: {:?}", checked.kind(items[0]));
    };
    assert!(is_int(&checked, *target, IntTy::I64));
}

/// operators §2: `as_mut` through a `&` lends no `&mut`, and a value that
/// is no transparent structured type has no parts to lend.
#[test]
fn as_mut_through_a_shared_reference_and_as_ref_of_a_primitive_are_refused() {
    let (checked, _) = Fixture::new().script("let p = (1, 2); let q = &p; let r = q.as_mut(); 0");
    assert!(matches!(checked.refusals()[..], [RefusalKind::MutLendThroughShared { .. }]), "{:?}", checked.refusals());
    for source in ["let x = 1; let y = &x; let r = y.as_ref(); 0", "let x = 1; let r = x.as_ref(); 0"] {
        let (checked, _) = Fixture::new().script(source);
        assert!(matches!(checked.refusals()[..], [RefusalKind::LendsNoParts { .. }]), "{source}: {:?}", checked.refusals());
    }
}

/// types §4, ownership §5: an array whose elements are references lends
/// each reborrowed at the weaker mutability: `as_ref` of `[&mut T; n]` is
/// `[&T; n]`, and `as_mut` of `[&T; n]` is `[&T; n]`.
#[test]
fn a_lend_of_an_array_of_references_reborrows_each_at_the_weaker_mutability() {
    for source in [
        "let x = 1; let y = 2; let a = [&mut x, &mut y]; let r = a.as_ref(); 0",
        "let x = 1; let y = 2; let a = [&x, &y]; let r = a.as_mut(); 0",
    ] {
        let fx = Fixture::new();
        let names = fx.names.clone();
        let (checked, script) = fx.script(source);
        checked.admitted();
        let TyKind::Array(element, _, ()) = checked.kind(binding(&checked, &script, &names, "r")) else {
            panic!("{source}: an array");
        };
        let TyKind::Ref(Mutability::Shared, target) = checked.kind(*element) else {
            panic!("{source}: each element a `&`, not {:?}", checked.kind(*element));
        };
        assert!(is_int(&checked, *target, IntTy::I64), "{source}");
    }
}

/// functions §2: `&f` of a generalized binding lends the value its name
/// reads at that use's instance, so two borrows reach two signatures.
#[test]
fn a_generalized_binding_is_lent_at_each_uses_signature() {
    let mut fx = Fixture::new();
    let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    for (name, ty) in [("on_int", i64_ty), ("on_bool", bool_ty)] {
        let taken = fx.pure(&[ty], ty);
        let lent = fx.reference(Mutability::Shared, taken);
        let applies = fx.pure(&[lent], ty);
        fx.declare(name, applies, &[]);
    }
    let (checked, _) = fx.script("let f = |x| -> x; let a = on_int(&f); let b = on_bool(&f); 0");
    checked.admitted();
    assert_eq!(checked.typeck.instances.len(), 2);
}

// Transparency at a reach, and function values (types §1-2)

/// types §1: a store through a reference reaches some place of its
/// target's type, not which one, so a lambda stored through a `&mut` to a
/// binding holding another is not one of the binding's function values.
#[test]
fn a_lambda_stored_through_a_mut_reference_to_another_is_refused() {
    let fx = Fixture::new();
    let (checked, _) = fx.script("let f = |v| -> v; let m = &mut f; *m = |v| -> 1; 0");
    assert!(
        matches!(checked.refusals()[..], [RefusalKind::NotOneOfFunctionValues { value: crate::ty::FnValue::Lambda(_) }]),
        "{:?}",
        checked.refusals()
    );
}

/// types §1-2: the binding's own value stored back through the reference
/// is one of its function values.
#[test]
fn a_bindings_own_value_stored_through_a_mut_reference_is_admitted() {
    let fx = Fixture::new();
    let (checked, _) = fx.script("let f = |v| -> v; let m = &mut f; *m = f; 0");
    checked.admitted();
}

/// types §1: a store into a field through a reference does not grow the
/// target's object by a field it lacks.
#[test]
fn a_field_stored_through_a_mut_reference_does_not_grow_its_target() {
    let fx = Fixture::new();
    let (checked, _) = fx.script("let a = { x: 1, }; let r = &mut a; r.y = 2; 0");
    assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque]);
}

/// types §1: a read of a field past a `*` reaches the target through the
/// reference, and does not grow it either.
#[test]
fn a_field_read_past_a_deref_does_not_grow_its_target() {
    let fx = Fixture::new();
    let (checked, _) = fx.script("let a = { x: 1, }; let r = &a; (*r).y");
    assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque]);
}

/// types §1: the same store by name grows the binding's object.
#[test]
fn a_field_stored_by_name_grows_the_bindings_object() {
    let fx = Fixture::new();
    let (x, y) = (fx.name("x"), fx.name("y"));
    let names = fx.names.clone();
    let (checked, script) = fx.script("let a = { x: 1, }; a.y = 2; 0");
    checked.admitted();
    let TyKind::Object { fields, label: () } = checked.kind(binding(&checked, &script, &names, "a")) else {
        panic!("not a structural object");
    };
    let names: Vec<Astr> = fields.iter().map(|(name, _)| *name).collect();
    let mut expected = vec![x, y];
    expected.sort();
    assert_eq!(names, expected);
}

/// types §1, patterns §1: a pattern past a reference matches the target
/// through it, so a field the pattern names and the target lacks is
/// refused rather than grown; one the target has is bound.
#[test]
fn a_pattern_through_a_reference_does_not_grow_its_target() {
    let fx = Fixture::new();
    let (checked, _) = fx.script("let o = { x: 1, }; let r = &o; match r { { y, } => 1, }");
    assert_eq!(checked.refusals().first(), Some(&&RefusalKind::GrowsOpaque), "{:?}", checked.typeck.refusals);
    let fx = Fixture::new();
    let (checked, _) = fx.script("let o = { x: 1, y: 2, }; let r = &o; match r { { x, } => *x, }");
    checked.admitted();
}

/// types §1, §3: a lambda entering a declared function type's parameter
/// is coerced to it, its function value forgotten there.
#[test]
fn a_lambda_passed_to_a_declared_parameter_is_admitted() {
    for source in ["apply(|x| -> 1)", "let f = |x| -> x; apply(f)"] {
        let mut fx = Fixture::new();
        declare_apply(&mut fx);
        let (checked, _) = fx.script(source);
        checked.admitted();
    }
}

/// types §2: a join of two lambdas may be either.
#[test]
fn two_lambdas_joined_by_if_are_either_function_value() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let script = acvus_ast::parse_script(&fx.names, "let h = if $c { |v| -> v } else { |v| -> true }; h(false)").expect("the test's source parses");
    let decls = Declarations::verify(fx.raw).expect("the test's declarations verify");
    let interfaces = super::Interfaces::new();
    let mut cx = super::ctxt::Ctxt::new(&fx.names, &decls, &interfaces, super::scope::Run::Script);
    super::stmt::check_body(&mut cx, &script.stmts, script.tail.as_deref());
    super::generalize::generalize_all(&mut cx);
    super::decision::solve(&mut cx);
    assert!(cx.out.refusals.is_empty(), "refused: {:?}", cx.out.refusals);

    let Some(Stmt::LetBind { binder, .. }) = script.stmts.first() else {
        panic!("the script binds h");
    };
    let h = cx.infer.shallow(cx.out.types[&binder.id]);
    let TyKind::Fn { values, .. } = cx.infer.types().kind(h).clone() else {
        panic!("h is a function type");
    };
    let lambdas: std::collections::BTreeSet<_> = cx.out.lambdas.iter().map(|lambda| crate::ty::FnValue::Lambda(lambda.site)).collect();
    assert_eq!(lambdas.len(), 2);
    assert_eq!(cx.infer.values(values), Some(crate::ty::FnValues::Of(lambdas)));
}

/// types §1: every step after a reference on a place's path is reached
/// through it, however far the path goes on: a store or a read down a
/// field or an element past an auto-deref grows nothing.
#[test]
fn a_path_past_a_reference_does_not_grow_its_target() {
    for source in [
        "let a = { b: { x: 1, }, }; let r = &mut a; r.b.y = 2; 0",
        "let a = [{ x: 1, }]; let r = &mut a; r[0].y = 2; 0",
        "let a = { b: { x: 1, }, }; let r = &a; r.b.y",
    ] {
        let fx = Fixture::new();
        let (checked, _) = fx.script(source);
        assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque], "{source}");
    }
}

/// types §1: the same paths by name grow the binding's type.
#[test]
fn a_path_by_name_grows_the_bindings_type() {
    for source in ["let a = { b: { x: 1, }, }; a.b.y = 2; 0", "let a = [{ x: 1, }]; a[0].y = 2; 0"] {
        let fx = Fixture::new();
        let (checked, _) = fx.script(source);
        checked.admitted();
    }
}

/// types §4-6: a `return` in one branch decides nothing of the `if`'s
/// type, in either order, so a later use fixes it.
#[test]
fn a_branch_that_returns_leaves_the_if_to_its_other_branch_in_either_order() {
    for source in [
        "let x; let y = if $c { return 1u8 } else { x }; x = 2u8; y",
        "let x; let y = if $c { x } else { return 1u8 }; x = 2u8; y",
    ] {
        let mut fx = Fixture::new();
        let bool_ty = fx.ty(TyKind::Bool);
        fx.input("c", bool_ty);
        let names = fx.names.clone();
        let (checked, script) = fx.script(source);
        checked.admitted();
        assert!(is_int(&checked, binding(&checked, &script, &names, "y"), IntTy::U8), "{source}");
    }
}

/// types §1, functions §2: a generalized lambda storing through a reference
/// it is given reaches the caller's place through it at each use, so it
/// does not grow the caller's object, as the same store written directly
/// does not; by name, into its own local, it grows.
#[test]
fn a_generalized_lambda_stores_through_a_reference_as_written_directly() {
    let fx = Fixture::new();
    let (checked, _) = fx.script("let set = |r| -> { r.y = 2; 0 }; let a = { x: 1, }; set(&mut a)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque]);
    let fx = Fixture::new();
    let (checked, _) = fx.script("let set = |v| -> { let s = { a: v, }; let p = &mut s; p.a.y = 2; 0 }; set({ x: 1, })");
    assert_eq!(checked.refusals(), vec![&RefusalKind::GrowsOpaque], "a step made where the lambda is written, read at the use");
    let fx = Fixture::new();
    let (checked, _) = fx.script("let set = |v| -> { let o = { x: v, }; o.y = 2; o }; let a = set(1); a.y");
    checked.admitted();
}

/// ownership §5, operators §6: a place reached through a shared reference
/// is neither stored into nor lent `&mut`, a reference part past it
/// included.
#[test]
fn a_place_past_a_shared_reference_is_not_written_or_lent_mut() {
    for (source, refused) in [
        ("let o = { x: 1, }; let r = &o; r.x = 2; 0", RefusalKind::StoreThroughShared),
        ("let o = { x: 1, }; let r = &o; let m = &mut r.x; *m = 2; 0", RefusalKind::LendMutThroughShared),
        ("let v = 1; let o = { b: &mut v, }; let r = &o; let m = &mut r.b; *m = 2; 0", RefusalKind::LendMutThroughShared),
    ] {
        let fx = Fixture::new();
        let (checked, _) = fx.script(source);
        assert!(checked.refusals().contains(&&refused), "{source}: {:?}", checked.refusals());
    }
    for source in [
        "let o = { x: 1, }; let r = &mut o; r.x = 2; let m = &mut r.x; *m = 3; 0",
        "let o = { x: 1, }; let r = &o; let s = &r.x; *s",
    ] {
        let fx = Fixture::new();
        let (checked, _) = fx.script(source);
        checked.admitted();
    }
}


/// extern-fn §1: an extern fn is called, never taken as a value. `len`
/// named where no call of it stands is refused there, and nothing else is.
#[test]
fn an_extern_fn_taken_as_a_value_is_refused() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let shared = fx.reference(Mutability::Shared, i64_ty);
    let len_ty = fx.pure(&[shared], i64_ty);
    let len = fx.declare("len", len_ty, &[]);
    let source = "let xs = 1; let f = len; f(&xs)";
    let (checked, _) = fx.script(source);
    assert_eq!(checked.refusals(), vec![&RefusalKind::ExternAsValue { name: len }]);
    let start = source.find("= len;").expect("the source names `len`") + "= ".len();
    assert_eq!(checked.typeck.refusals[0].span, acvus_ast::Span { start, end: start + "len".len() }, "refused at the name");
}

/// extern-fn §1: the value use is what is refused, so a bare name reaching
/// two declarations is refused as one used as a value, by the name as
/// written, and not as several declarations.
#[test]
fn a_name_reaching_several_extern_fns_taken_as_a_value_is_refused_by_its_written_name() {
    let mut fx = Fixture::new();
    let i64_ty = fx.int(IntTy::I64);
    let shared = fx.reference(Mutability::Shared, i64_ty);
    let len_ty = fx.pure(&[shared], i64_ty);
    fx.overload("a", "len", len_ty);
    fx.overload("b", "len", len_ty);
    let written = QualifiedRef::root(fx.name("len"));
    let source = "let xs = 1; let f = len; f(&xs)";
    let (checked, _) = fx.script(source);
    assert_eq!(checked.refusals(), vec![&RefusalKind::ExternAsValue { name: written }]);
    let start = source.find("= len;").expect("the source names `len`") + "= ".len();
    assert_eq!(checked.typeck.refusals[0].span, acvus_ast::Span { start, end: start + "len".len() }, "refused at the name");
}

/// An extern type `X` carrying one identity and `Y` carrying none, with,
/// where `ancestor`, the cast `X -> Y`; `make() -> X<I>`, whose identity
/// only its result holds; `keep(x: X<I>) -> X<I>` and `join(a: X<I>, b:
/// X<I>) -> X<I>`, which give their result their parameters' identity;
/// `each(f: fn(i64) -> i64) -> i64`, whose closure may hold a `&mut`; and
/// `apply(f: fn(i64) -> X<I>) -> X<I>`. The fixture, and `Y`'s name.
fn identity_fixture(ancestor: bool) -> (Fixture, QualifiedRef) {
    identity_fixture_in(Fixture::new(), ancestor)
}

/// The identity fixture declared into `fx`.
fn identity_fixture_in(mut fx: Fixture, ancestor: bool) -> (Fixture, QualifiedRef) {
    let (x, y) = (QualifiedRef::root(fx.name("X")), QualifiedRef::root(fx.name("Y")));
    for (id, carries_identity) in [(x, true), (y, false)] {
        fx.raw.user_types.push(UserTypeDecl { id, type_params: Box::new([]), carries_effect: false, carries_identity, lifetimes: 0 });
    }
    let x_i = fx.ty(TyKind::UserDefined {
        id: x,
        type_args: Box::new([]),
        effect: None,
        identity: Some(IdentityTerm::Var(0)),
        marked: MarkTerm::Known(Mark::Unmarked),
    });
    let y_ty = fx.ty(TyKind::UserDefined { id: y, type_args: Box::new([]), effect: None, identity: None, marked: MarkTerm::Known(Mark::Unmarked) });
    let i64_ty = fx.int(IntTy::I64);
    let make = fx.pure(&[], x_i);
    fx.declare("make", make, &[]);
    let keep = fx.pure(&[x_i], x_i);
    fx.declare("keep", keep, &[]);
    let join = fx.pure(&[x_i, x_i], x_i);
    fx.declare("join", join, &[]);
    let holding_mut = fx.ty(TyKind::Fn {
        params: Box::new([i64_ty]),
        ret: i64_ty,
        effect: EffectTerm::Known(crate::ty::Effect::pure()),
        flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
        calls: Calls::Many,
        holds: Holds::Mut,
        values: (),
    });
    let each = fx.pure(&[holding_mut], i64_ty);
    fx.declare("each", each, &[]);
    let making = fx.pure(&[i64_ty], x_i);
    let apply = fx.pure(&[making], x_i);
    fx.declare("apply", apply, &[]);
    if ancestor {
        let widening = fx.pure(&[x_i], y_ty);
        let to_y = fx.declare("to_y", widening, &[]);
        fx.raw.casts.push(CastRule { from: x_i, to: y_ty, via: to_y });
    }
    (fx, y)
}

/// Whether `ty` is the extern type `named`.
fn is_named(checked: &Checked, ty: Ty, named: QualifiedRef) -> bool {
    matches!(checked.kind(ty), TyKind::UserDefined { id, .. } if *id == named)
}

/// Whether the one refusal is two identities meeting at a declaration's
/// parameter, a node that climbs to no ancestor, so the two stay two
/// heads (types §3).
fn refused_as_two_heads(checked: &Checked) -> bool {
    matches!(checked.refusals()[..], [RefusalKind::Mismatch(FrozenMismatch { mismatch: Mismatch::Heads { .. }, .. })])
}

/// `source` checked against the identity fixture, `ancestor` as it says,
/// and the type of its binding `name`.
fn identity_binding(ancestor: bool, source: &str, name: &str) -> (Checked, Ty, QualifiedRef) {
    let (fx, y) = identity_fixture(ancestor);
    let names = fx.names.clone();
    let (checked, script) = fx.script(source);
    let ty = binding(&checked, &script, &names, name);
    (checked, ty, y)
}

/// types §3: each call of a declaration whose result carries an identity
/// its parameters do not give makes a fresh one. The values of two calls
/// are of two identities: at a declaration's parameter, where the second
/// cannot climb, they are refused, as two heads with the ancestor and as no
/// conversion without it; stored into one binding, they meet at the
/// ancestor.
#[test]
fn two_calls_give_two_identities() {
    let (checked, _) = identity_fixture(true).0.script("join(make(), make())");
    assert!(refused_as_two_heads(&checked), "refused: {:?}", checked.refusals());
    let (checked, _) = identity_fixture(false).0.script("join(make(), make())");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoConversion]);

    let (checked, a, y) = identity_binding(true, "let a = make(); a = make(); a", "a");
    checked.admitted();
    assert!(is_named(&checked, a, y), "`a` is at the ancestor");
}

/// types §3: a declaration gives its result a parameter's identity, so one
/// value's identity kept through `keep` meets itself.
#[test]
fn an_identity_kept_through_a_parameter_is_one() {
    let (fx, y) = identity_fixture(true);
    let (checked, script) = fx.script("let a = make(); join(a, keep(a))");
    checked.admitted();
    assert!(!is_named(&checked, tail(&checked, &script), y), "the join keeps `X`");
}

/// types §3: an iteration of a loop is one instantiation of its body, and
/// an identity made in it never enters what was made outside it: stored
/// into a binding the loop's body did not declare, the value takes its
/// nearest ancestor carrying no identity, and is refused where there is
/// none.
#[test]
fn an_identity_made_in_a_loop_takes_its_ancestor_outside_it() {
    let source = "let last = make(); for i in 0..3 { last = make(); } last";
    let (checked, last, y) = identity_binding(true, source, "last");
    checked.admitted();
    assert!(is_named(&checked, last, y), "`last` is at the ancestor");
    let (checked, _, _) = identity_binding(false, source, "last");
    assert_eq!(checked.refusals(), vec![&RefusalKind::IdentityEscapes]);
}

/// types §3: a call of a lambda is one instantiation of its body, and a
/// captured binding was made outside it: an identity made in the lambda
/// stored into one takes its ancestor, as a loop's does.
#[test]
fn an_identity_made_in_a_lambda_takes_its_ancestor_in_a_capture() {
    let source = "let v = make(); each(|i| -> { v = make(); i }); v";
    let (checked, v, y) = identity_binding(true, source, "v");
    checked.admitted();
    assert!(is_named(&checked, v, y), "`v` is at the ancestor");
    let (checked, _, _) = identity_binding(false, source, "v");
    assert_eq!(checked.refusals(), vec![&RefusalKind::IdentityEscapes]);
}

/// types §3: a lambda whose value is not generalized gives its result's
/// identity made in its body nowhere outside it: at `apply`'s result,
/// whose identity the caller's call holds, it escapes, refused without an
/// ancestor.
#[test]
fn an_identity_made_in_a_lambda_not_generalized_escapes_at_its_result() {
    let (checked, _) = identity_fixture(false).0.script("apply(|i| -> make())");
    assert_eq!(checked.refusals(), vec![&RefusalKind::IdentityEscapes]);
}

/// types §3: a let-bound lambda's result identity made in its body leaves
/// as its result, made fresh at each call as a declaration's is: one call's
/// value kept through `keep` meets itself, and two calls' values are two
/// identities.
#[test]
fn a_let_bound_lambdas_result_identity_is_fresh_at_each_call() {
    let (fx, y) = identity_fixture(true);
    let (checked, script) = fx.script("let f = |i| -> make(); let a = f(1); join(a, keep(a))");
    checked.admitted();
    assert!(!is_named(&checked, tail(&checked, &script), y), "the join keeps `X`");
    let (checked, _) = identity_fixture(true).0.script("let f = |i| -> make(); join(f(1), f(2))");
    assert!(refused_as_two_heads(&checked), "refused: {:?}", checked.refusals());
}

/// types §3: a call of a unit is an instantiation of its body, and its
/// result's identity made there is made fresh at each call, as a
/// declaration's is: two calls' values are two identities, and one call's
/// value kept through `keep` meets itself.
#[test]
fn a_units_result_identity_is_fresh_at_each_call() {
    let unit = [(&["lib"][..], "m", "make()")];
    let (checked, _, _) = identity_fixture(true).0.script_with_units(&unit, "join(lib::m(), lib::m())");
    assert!(refused_as_two_heads(&checked), "refused: {:?}", checked.refusals());
    let (checked, _, _) = identity_fixture(false).0.script_with_units(&unit, "join(lib::m(), lib::m())");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoConversion]);

    let (fx, y) = identity_fixture(true);
    let names = fx.names.clone();
    let (checked, script, _) = fx.script_with_units(&unit, "let a = lib::m(); a = lib::m(); a");
    checked.admitted();
    assert!(is_named(&checked, binding(&checked, &script, &names, "a"), y), "`a` is at the ancestor");

    let (fx, y) = identity_fixture(true);
    let (checked, script, _) = fx.script_with_units(&unit, "let a = lib::m(); join(a, keep(a))");
    checked.admitted();
    assert!(!is_named(&checked, tail(&checked, &script), y), "the join keeps `X`");
}

/// context §3: a context's identity is fresh, one per context, so two
/// contexts' values never meet, and one context's value read twice is of
/// one identity.
#[test]
fn a_contexts_identity_is_its_own() {
    let with_contexts = |ancestor: bool| {
        let (mut fx, y) = identity_fixture(ancestor);
        let x = QualifiedRef::root(fx.name("X"));
        let x_i = fx.ty(TyKind::UserDefined {
            id: x,
            type_args: Box::new([]),
            effect: None,
            identity: Some(IdentityTerm::Var(0)),
            marked: MarkTerm::Known(Mark::Unmarked),
        });
        fx.context("p", x_i);
        fx.context("q", x_i);
        (fx, y)
    };
    let (checked, _) = with_contexts(true).0.script("join(@p, @q)");
    assert!(refused_as_two_heads(&checked), "refused: {:?}", checked.refusals());
    let (checked, _) = with_contexts(false).0.script("join(@p, @q)");
    assert_eq!(checked.refusals(), vec![&RefusalKind::NoConversion]);

    let (fx, y) = with_contexts(true);
    let (checked, script) = fx.script("join(@p, @p)");
    checked.admitted();
    assert!(!is_named(&checked, tail(&checked, &script), y), "the join keeps `X`");
}

/// context §3, types §3: a context's identity is made once, at the
/// declarations, outside every body, so a unit's result holding it is not
/// made fresh at its calls: a unit returning `@p`'s value gives the caller
/// the identity of the caller's own `@p`, and two units reading `@p` give
/// one identity.
#[test]
fn a_contexts_identity_is_one_across_units() {
    let with_context = || {
        let (mut fx, _) = identity_fixture(false);
        let x = QualifiedRef::root(fx.name("X"));
        let x_i = fx.ty(TyKind::UserDefined {
            id: x,
            type_args: Box::new([]),
            effect: None,
            identity: Some(IdentityTerm::Var(0)),
            marked: MarkTerm::Known(Mark::Unmarked),
        });
        fx.context("p", x_i);
        fx
    };
    let (checked, _, _) = with_context().script_with_units(&[(&["lib"][..], "g", "@p")], "join(lib::g(), @p)");
    checked.admitted();
    let units = [(&["lib"][..], "g", "@p"), (&["lib"][..], "h", "@p")];
    let (checked, _, _) = with_context().script_with_units(&units, "join(lib::g(), lib::h())");
    checked.admitted();
}

/// graph, early cutoff: an interface is the same as another but for the
/// identities its check's bodies made. Each check allocates those anew, so
/// the canonical form numbers them by the order it first reaches them; an
/// identity made outside every body, a context's, is the same storage in
/// every check under one verification of the declarations, and keeps its
/// id. `$x` is of the identity `make` gave in the body, and the result
/// holds none, so the interface holds it known.
#[test]
fn an_interface_is_the_same_but_for_the_identities_its_bodies_made() {
    let names = Interner::new();
    let with_context = || {
        let (mut fx, _) = identity_fixture_in(Fixture::with_names(names.clone()), false);
        let x = QualifiedRef::root(fx.name("X"));
        let x_i = fx.ty(TyKind::UserDefined {
            id: x,
            type_args: Box::new([]),
            effect: None,
            identity: Some(IdentityTerm::Var(0)),
            marked: MarkTerm::Known(Mark::Unmarked),
        });
        fx.context("p", x_i);
        fx
    };
    let body_made = "let a = join(make(), $x); 1";
    let [one, other] = &with_context().interfaces(&[body_made, body_made])[..] else {
        panic!("two interfaces");
    };
    assert!(one.same(other), "one source checked twice is the same");

    let [one, other] = &with_context().interfaces(&["@p", "@p"])[..] else {
        panic!("two interfaces");
    };
    assert!(one.same(other), "a context's identity is one under one verification");
    let [elsewhere] = &with_context().interfaces(&["@p"])[..] else {
        panic!("one interface");
    };
    assert!(!one.same(elsewhere), "two verifications make two identities of the context");
}

// -- what the run declares: its parameters and the contexts each body names ---

/// programs §1-§2, §4: the run's parameters are the inputs its source
/// reads, in the order of the unit's parameters, the text of their names,
/// whatever order the source reads them in; an input given and not read is
/// none.
#[test]
fn the_runs_parameters_are_the_inputs_it_reads_by_name() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    for input in ["b", "a", "c"] {
        fx.input(input, int);
    }
    let names = fx.names.clone();
    let (checked, _) = fx.script("($b, $a)");
    checked.admitted();
    let params: Vec<&str> = checked.typeck.inputs.iter().map(|input| names.resolve(input.name)).collect();
    assert_eq!(params, ["a", "b"]);
}

/// context §3: each body names its own contexts, the run's outside its
/// lambdas and a lambda's in its body, each once, at its declared type.
#[test]
fn each_body_names_its_own_contexts_once() {
    let mut fx = Fixture::new();
    let int = fx.int(IntTy::I64);
    fx.context("x", int);
    fx.context("y", int);
    let names = fx.names.clone();
    let (checked, _) = fx.script("let a = @x; let f = |z| -> @y; @x");
    checked.admitted();
    let named = |body: &[(QualifiedRef, Ty)]| -> Vec<String> {
        body.iter().map(|(context, ty)| format!("{}: {:?}", names.resolve(context.name), checked.kind(*ty))).collect()
    };
    assert_eq!(named(&checked.typeck.contexts[&None]), ["x: Int(I64)"]);
    let lambdas: Vec<Vec<String>> = checked.typeck.contexts.iter().filter(|(body, _)| body.is_some()).map(|(_, body)| named(body)).collect();
    assert_eq!(lambdas, [vec!["y: Int(I64)".to_string()]]);
}

// -- a value of type `!`: what a diverging form decides (types §4-6) -------
//
// `!` is the type of no value. A form whose value is `!`, a `return`, a
// block whose last statement is `break` or `continue`, or a call of a
// declaration whose declared result is `!`, decides nothing of its own
// type: its type is what its uses decide, and `!` where nothing does
// (types §4, §5). `!` says nothing of a type: a decision that reads it
// decides nothing from it and refuses nothing for it, and at a reach every
// candidate matches it at an argument position, the receiver included,
// and it is no type a result is taken at (types §6). So whether a program
// is admitted does not depend on which of these forms gives the `!`, nor
// on the order of a call's arguments.

/// `statements`, each `{X}` a form whose value is `!`, then the tail `0`, in
/// each way such a form is written: directly, `(return 1)`; where two arms
/// that both return meet, `(if $c { return 1 } else { return 2 })`; read
/// from a binding given one, `let a = return 1;` and `a`; and inside a
/// `while`, where an arm that breaks meets one that continues,
/// `(if $c { break; } else { continue; })`.
fn diverging_forms(statements: &str) -> [String; 4] {
    let at = |value: &str| statements.replace("{X}", value);
    [
        format!("{} 0", at("(return 1)")),
        format!("{} 0", at("(if $c { return 1 } else { return 2 })")),
        format!("let a = return 1; {} 0", at("a")),
        format!("while $c {{ {} }} 0", at("(if $c { break; } else { continue; })")),
    ]
}

/// `statements` in the body of the lambda `k`, then its tail `y`, in each
/// form `diverging_forms` writes, each `return` leaving `k` with `y`; the
/// run calls `k(3)`.
fn diverging_forms_in_a_lambda(statements: &str) -> [String; 4] {
    let at = |value: &str| statements.replace("{X}", value);
    [
        format!("let k = |y| -> {{ {} y }}; k(3)", at("(return y)")),
        format!("let k = |y| -> {{ {} y }}; k(3)", at("(if $c { return y } else { return y })")),
        format!("let k = |y| -> {{ let a = return y; {} y }}; k(3)", at("a")),
        format!("let k = |y| -> {{ while $c {{ {} }} y }}; k(3)", at("(if $c { break; } else { continue; })")),
    ]
}

/// `statements`, each `{X}` a call of `abort`, whose declared result is
/// `!`, then the tail `0`, in each way one is written: directly, `abort()`;
/// where two arms that both call it meet,
/// `(if $c { abort() } else { abort() })`; and read from a binding given
/// one, `let a = abort();` and `a`.
fn aborting_forms(statements: &str) -> [String; 3] {
    let at = |value: &str| statements.replace("{X}", value);
    [
        format!("{} 0", at("abort()")),
        format!("{} 0", at("(if $c { abort() } else { abort() })")),
        format!("let a = abort(); {} 0", at("a")),
    ]
}

/// `statements` in the body of the lambda `k`, then its tail `y`, in each
/// form `aborting_forms` writes; the run calls `k(3)`.
fn aborting_forms_in_a_lambda(statements: &str) -> [String; 3] {
    let at = |value: &str| statements.replace("{X}", value);
    [
        format!("let k = |y| -> {{ {} y }}; k(3)", at("abort()")),
        format!("let k = |y| -> {{ {} y }}; k(3)", at("(if $c { abort() } else { abort() })")),
        format!("let k = |y| -> {{ let a = abort(); {} y }}; k(3)", at("a")),
    ]
}

/// What this section's programs read: the input `$c: bool`, `abort() -> !`,
/// `g(i64) -> i64`, `h(&i64) -> i64`, and `core::eq`, `core::cmp` and
/// `core::add` each at `i64` alone.
fn never_fixture() -> Fixture {
    let mut fx = Fixture::new();
    let (bool_ty, i64_ty) = (fx.ty(TyKind::Bool), fx.int(IntTy::I64));
    fx.input("c", bool_ty);
    let never = fx.ty(TyKind::Never);
    let aborts = fx.pure(&[], never);
    fx.declare("abort", aborts, &[]);
    let by_value = fx.pure(&[i64_ty], i64_ty);
    fx.declare("g", by_value, &[]);
    let shared = fx.reference(Mutability::Shared, i64_ty);
    let lent = fx.pure(&[shared], i64_ty);
    fx.declare("h", lent, &[]);
    let (eq, cmp, add) = (fx.raw.core.eq, fx.raw.core.cmp, fx.raw.core.add);
    let equality = fx.pure(&[shared, shared], bool_ty);
    fx.lang_item(eq, &[equality]);
    let order = fx.pure(&[shared, shared], i64_ty);
    fx.lang_item(cmp, &[order]);
    fx.arithmetic(add, &[TyKind::Int(IntTy::I64)]);
    fx
}

/// `never_fixture` with `f` declared over `(u8, u8)` and over `(bool, bool)`,
/// each returning a `u8`, in two namespaces a bare `f` reaches both of; and
/// the `(u8, u8)` one.
fn with_two_overloads_of_f() -> (Fixture, QualifiedRef) {
    let mut fx = never_fixture();
    let (u8_ty, bool_ty) = (fx.int(IntTy::U8), fx.ty(TyKind::Bool));
    let of_bytes = fx.pure(&[u8_ty, u8_ty], u8_ty);
    let bytes = fx.overload("n0", "f", of_bytes);
    let of_bools = fx.pure(&[bool_ty, bool_ty], u8_ty);
    fx.overload("n1", "f", of_bools);
    (fx, bytes)
}

/// The refusals of `source` checked over `fixture`'s declarations, each
/// written, and the kinds among them.
fn never_verdict(fixture: Fixture, source: &str) -> (Vec<String>, std::collections::HashSet<std::mem::Discriminant<RefusalKind>>) {
    let names = fixture.names.clone();
    let (checked, _) = fixture.script(source);
    let written = checked.typeck.refusals.iter().map(|refusal| refusal.kind.written(&names)).collect();
    let kinds = checked.typeck.refusals.iter().map(|refusal| std::mem::discriminant(&refusal.kind)).collect();
    (written, kinds)
}

/// Each of `sources` admitted over the declarations `fixture` makes; the
/// failure names every one refused, with its refusals.
fn assert_each_admitted(fixture: impl Fn() -> Fixture, sources: &[String]) {
    let refused: Vec<String> = sources
        .iter()
        .filter_map(|source| {
            let (written, _) = never_verdict(fixture(), source);
            (!written.is_empty()).then(|| format!("{source}\n    refused: {written:?}"))
        })
        .collect();
    assert!(refused.is_empty(), "each is admitted, and these are refused:\n{}", refused.join("\n"));
}

/// Each of `sources` given one verdict over the declarations `fixture`
/// makes: all admitted, or all refused by the same kinds of refusal.
fn assert_one_verdict(fixture: impl Fn() -> Fixture, sources: &[String]) {
    let verdicts: Vec<_> = sources.iter().map(|source| (source, never_verdict(fixture(), source))).collect();
    let shown = verdicts
        .iter()
        .map(|(source, (written, _))| format!("{source}\n    refused: {written:?}"))
        .collect::<Vec<_>>()
        .join("\n");
    let alike = verdicts.windows(2).all(|pair| pair[0].1.1 == pair[1].1.1);
    assert!(alike, "each is admitted, or each refused by the same kinds:\n{shown}");
}

/// Each of `sources` refused over the declarations `fixture` makes, all by
/// the same kinds of refusal.
fn assert_each_refused_alike(fixture: impl Fn() -> Fixture, sources: &[String]) {
    assert_one_verdict(&fixture, sources);
    let admitted: Vec<&String> = sources.iter().filter(|source| never_verdict(fixture(), source).0.is_empty()).collect();
    assert!(admitted.is_empty(), "each is refused, and these are admitted:\n{admitted:#?}");
}

/// `sources` checked over `fixture`'s declarations, each admitted, and its
/// every reached call but those of `abort`, which give the `!`, reaching
/// the declaration `fixture` names.
fn assert_each_reaches(fixture: impl Fn() -> (Fixture, QualifiedRef), sources: &[String]) {
    let failed: Vec<String> = sources
        .iter()
        .filter_map(|source| {
            let (fx, wanted) = fixture();
            let names = fx.names.clone();
            let abort = QualifiedRef::root(names.intern("abort"));
            let (checked, _) = fx.script(source);
            let refused: Vec<String> = checked.typeck.refusals.iter().map(|refusal| refusal.kind.written(&names)).collect();
            let reached: Vec<QualifiedRef> =
                checked.typeck.reached.values().map(|chosen| chosen.declaration).filter(|declaration| *declaration != abort).collect();
            let shown = |declaration: &QualifiedRef| crate::ty::print::qualified(&names, *declaration);
            let fits = refused.is_empty() && reached == [wanted];
            (!fits).then(|| format!("{source}\n    refused: {refused:?}; reached: {:?}", reached.iter().map(shown).collect::<Vec<_>>()))
        })
        .collect();
    assert!(failed.is_empty(), "each is admitted and reaches the one declaration; these are not:\n{}", failed.join("\n"));
}

/// The sources of every one of `templates` in every form `forms` writes.
fn every_form<const FORMS: usize>(forms: fn(&str) -> [String; FORMS], templates: &[&str]) -> Vec<String> {
    templates.iter().flat_map(|template| forms(template)).collect()
}

/// The sources of every one of `templates` in every form `diverging_forms`
/// and `aborting_forms` write.
fn every_diverging_form(templates: &[&str]) -> Vec<String> {
    [every_form(diverging_forms, templates), every_form(aborting_forms, templates)].concat()
}

/// The sources of every one of `templates`, each `{X}` a `while`'s
/// condition or a `while let`'s source, in every form but the loop's: a
/// `break` or a `continue` there is refused (control-flow,
/// `break_or_continue_in_a_while_condition_is_refused`).
fn every_diverging_form_in_a_condition(templates: &[&str]) -> Vec<String> {
    let returning = templates.iter().flat_map(|template| diverging_forms(template).into_iter().take(3));
    returning.chain(every_form(aborting_forms, templates)).collect()
}

/// types §4-6, control-flow: a condition of a diverging form is decided by
/// the condition: `if X`, `while X` and `if !X` are admitted in every form,
/// `while X` in every form but the loop's.
#[test]
fn a_condition_of_a_diverging_form_is_decided_by_the_condition() {
    let mut sources = every_diverging_form(&["if {X} { 2 } else { 3 };", "if !{X} { 2 } else { 3 };"]);
    sources.extend(every_diverging_form_in_a_condition(&["while {X} { }"]));
    assert_each_admitted(never_fixture, &sources);
}

/// types §4-6, operators §2: an operand of `&&` or `||` of a diverging form,
/// on either side, is decided by the operator: admitted in every form.
#[test]
fn an_operand_of_and_or_or_of_a_diverging_form_is_decided_by_the_operator() {
    let templates = ["let b = {X} && true;", "let b = true && {X};", "let b = {X} || true;", "let b = true || {X};"];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, control-flow: a range's end of a diverging form is an integer
/// of the range's one width: admitted at either end in every form.
#[test]
fn a_range_end_of_a_diverging_form_is_decided_by_the_range() {
    let templates = ["for i in {X}..3 { }", "for i in 0..{X} { }"];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, operators §2, §4: an operand of `==`, `!=` or `<` of a
/// diverging form meets the other operand's type, on either side, and
/// compares through `core::eq` or `core::cmp` there: admitted in every form.
#[test]
fn a_comparison_operand_of_a_diverging_form_meets_the_other_operand() {
    let templates = [
        "let b = {X} == 1;",
        "let b = 1 == {X};",
        "let b = {X} != 1;",
        "let b = 1 != {X};",
        "let b = {X} < 1;",
        "let b = 1 < {X};",
    ];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, operators §2: an operand of `^`, `&`, `<<` or `>>` of a
/// diverging form, on either side, is admitted in every form: `^` and `&`
/// take it at the other operand's type, and a shift reads a `!` as no value.
#[test]
fn an_operand_of_the_languages_own_integer_operators_of_a_diverging_form_is_admitted() {
    let templates = [
        "let b = {X} ^ 1;",
        "let b = 1 ^ {X};",
        "let b = {X} & 1;",
        "let b = 1 & {X};",
        "let b = {X} << 1;",
        "let b = 1 << {X};",
        "let b = {X} >> 1;",
        "let b = 1 >> {X};",
    ];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6: an argument of a diverging form, by value to `g(i64)` or lent
/// to `h(&i64)`, the one declaration of its name, reaches it and enters its
/// parameter: admitted in every form.
#[test]
fn an_argument_of_a_diverging_form_reaches_the_one_declaration() {
    let templates = ["let r = g({X});", "let r = h(&{X});"];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §2-4: a part of a diverging form of a tuple, an object, a `Some` or
/// an array meets its sibling's part where the two values meet in an `if`'s
/// one slot: its type is what that meeting decides, so the two meet,
/// admitted in every form.
#[test]
fn a_constructor_part_of_a_diverging_form_meets_its_siblings_part() {
    let templates = [
        "let x = if $c { (0, 1) } else { (0, {X}) };",
        "let x = if $c { { a: 0, b: 1, } } else { { a: 0, b: {X}, } };",
        "let x = if $c { Some(1) } else { Some({X}) };",
        "let x = if $c { [0, 1] } else { [0, {X}] };",
    ];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, operators §3, §5, §6, control-flow: a field, an element, a
/// deref, a `?` and an `as` of a diverging form, used as a condition: what
/// each reads of a `!` decides nothing of the value it gives, which the
/// condition decides: admitted in every form.
#[test]
fn a_projection_of_a_diverging_form_is_decided_by_its_use() {
    let templates = [
        "if {X}.x { 2 } else { 3 };",
        "if {X}[0] { 2 } else { 3 };",
        "if *{X} { 2 } else { 3 };",
        "if {X}? { 2 } else { 3 };",
        "if {X} as i64 == 1 { 2 } else { 3 };",
    ];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, control-flow, patterns §4: a `for` over a diverging form and
/// a `match` on one bind names whose type their uses decide, here a
/// condition's `bool`: admitted in every form.
#[test]
fn a_for_head_or_a_scrutinee_of_a_diverging_form_binds_names_its_uses_decide() {
    let templates = ["for v in {X} { if v { } }", "match {X} { v => if v { 2 } else { 3 }, };"];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, control-flow: a diverging form standing as a statement, and
/// as the last statement of a block with no tail, which then decides nothing
/// of the block's type, is admitted in every form.
#[test]
fn a_diverging_form_as_a_statement_or_a_blocks_end_is_admitted() {
    let templates = ["{X};", "let x = if $c { 1 } else { {X}; };"];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4-6, functions §2: in a lambda's body a diverging form is what it
/// is in the run's: `if return y { 1 } else { 2 }` as the body, and a
/// condition, an argument, a tuple's part and a call of the overloaded `f`
/// in a block body, each `return` leaving the lambda, are admitted in every
/// form, a call of `abort` among them. The lambda named `f` is a binding,
/// which its call reads.
#[test]
fn a_diverging_form_in_a_lambda_body_is_admitted_as_in_the_run() {
    let mut sources = vec![
        "let f = |y| -> if return y { 1 } else { 2 }; f(3)".to_string(),
        "let f = |y| -> if (if $c { return y } else { return y }) { 1 } else { 2 }; f(3)".to_string(),
        "let f = |y| -> { let a = return y; if a { 1 } else { 2 } }; f(3)".to_string(),
    ];
    let templates = [
        "if {X} { 2 } else { 3 };",
        "let r = g({X});",
        "let x = if $c { (0, 1) } else { (0, {X}) };",
        "let r = f(2u8, {X});",
        "let r = f({X}, 2u8);",
    ];
    sources.extend(every_form(diverging_forms_in_a_lambda, &templates));
    sources.extend(every_form(aborting_forms_in_a_lambda, &templates));
    assert_each_admitted(|| with_two_overloads_of_f().0, &sources);
}

/// types §6: a reach reads a position of type `!` as one every candidate
/// matches, narrowing nothing there, so the order of the positions decides
/// nothing: with `f` over `(u8, u8)` and `(bool, bool)`, `f(2u8, X)` and
/// `f(X, 2u8)` each reach the `(u8, u8)` declaration, in every form.
#[test]
fn a_diverging_argument_narrows_no_overload_in_either_position() {
    let templates = ["let r = f(2u8, {X});", "let r = f({X}, 2u8);"];
    assert_each_reaches(with_two_overloads_of_f, &every_diverging_form(&templates));
}

/// types §6: with `o` over `i64` and over `bool`, each returning an `i64`,
/// and nothing else deciding, `o(X)` of a diverging `X` leaves both: it is
/// refused, by the same kinds of refusal in every form.
#[test]
fn a_diverging_argument_alone_before_two_overloads_is_refused_alike() {
    assert_each_refused_alike(with_two_overloads_of_o, &every_form(diverging_forms, &["let r = o({X});"]));
}

/// `never_fixture` with `o` declared over `i64` and over `bool`, each
/// returning an `i64`, in two namespaces a bare `o` reaches both of.
fn with_two_overloads_of_o() -> Fixture {
    let mut fx = never_fixture();
    let (i64_ty, bool_ty) = (fx.int(IntTy::I64), fx.ty(TyKind::Bool));
    let of_int = fx.pure(&[i64_ty], i64_ty);
    fx.overload("n0", "o", of_int);
    let of_bool = fx.pure(&[bool_ty], i64_ty);
    fx.overload("n1", "o", of_bool);
    fx
}

/// types §6: `o(X)`, `X` a call of `abort`, whose declared result is `!`,
/// narrows neither of `o`'s declarations, and nothing else decides: it is
/// refused, by the same kinds of refusal in the three forms.
#[test]
fn a_call_of_abort_alone_before_two_overloads_is_refused_alike() {
    assert_each_refused_alike(with_two_overloads_of_o, &every_form(aborting_forms, &["let r = o({X});"]));
}

/// types §5: a type variable nothing constrains is still `!`, and what reads
/// it decides nothing: `[] == []`, and a lambda whose parameter nothing
/// constrains reading a field, an element, a deref, a `?`, a `for` and an
/// `as` of it, are admitted.
#[test]
fn a_type_variable_nothing_constrains_is_never_and_read_as_no_value() {
    let sources = [
        "[] == []",
        "let f = |x| -> x.a; 0",
        "let f = |x| -> x[0]; 0",
        "let f = |x| -> *x; 0",
        "let f = |x| -> x?; 0",
        "let f = |x| -> { for v in x { } 0 }; 0",
        "let f = |x| -> x as i64; 0",
    ]
    .map(String::from);
    assert_each_admitted(never_fixture, &sources);
}

/// types §4-6: a `return` of a `return` is two diverging forms, each decided
/// by its uses: the inner one by the run's result it enters, the outer one
/// by the condition. Admitted in every form.
#[test]
fn a_return_of_a_return_is_decided_by_its_uses() {
    let sources = [
        "if (return (return 1)) { 2 } else { 3 }",
        "if (if $c { return (return 1) } else { return (return 2) }) { 2 } else { 3 }",
        "let a = return (return 1); if a { 2 } else { 3 }",
    ]
    .map(String::from);
    assert_each_admitted(never_fixture, &sources);
}

/// types §4-6, operators §2, §4: both operands of one operator of a
/// diverging form: nothing but each other decides them, so they are `!`,
/// which a comparison and the language's own operators read as no value:
/// admitted in every form.
#[test]
fn diverging_operands_on_both_sides_of_an_operator_are_admitted() {
    let templates = [
        "let b = {X} == {X};",
        "let b = {X} < {X};",
        "let b = {X} && {X};",
        "let b = {X} ^ {X};",
        "let b = {X} << {X};",
    ];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §6: `f(X, X)` of a diverging `X` narrows neither overload at
/// either position, and both give a `u8`, so both are left: refused, by the
/// same kinds of refusal in every form.
#[test]
fn diverging_arguments_at_every_position_leave_both_overloads_and_are_refused_alike() {
    assert_each_refused_alike(|| with_two_overloads_of_f().0, &every_form(diverging_forms, &["let r = f({X}, {X});"]));
}

/// types §6: `f(X, X)`, each `X` a call of `abort`, whose declared result is
/// `!`, narrows neither overload at either position, and both give a `u8`:
/// refused, by the same kinds of refusal in the three forms.
#[test]
fn calls_of_abort_at_every_position_leave_both_overloads_and_are_refused_alike() {
    assert_each_refused_alike(|| with_two_overloads_of_f().0, &every_form(aborting_forms, &["let r = f({X}, {X});"]));
}

/// types §6: a tuple is reached by its parts in order, and a part of type
/// `!` narrows nothing: with `p` over `((u8, u8))` and `((bool, bool))`,
/// `p((X, 2u8))` and `p((2u8, X))` each reach the `(u8, u8)` one, in every
/// form.
#[test]
fn a_diverging_tuple_part_narrows_no_overload_in_either_position() {
    let with_two_overloads_of_p = || {
        let mut fx = never_fixture();
        let (u8_ty, bool_ty) = (fx.int(IntTy::U8), fx.ty(TyKind::Bool));
        let bytes = fx.ty(TyKind::Tuple(Box::new([u8_ty, u8_ty]), ()));
        let of_bytes = fx.pure(&[bytes], u8_ty);
        let wanted = fx.overload("n0", "p", of_bytes);
        let bools = fx.ty(TyKind::Tuple(Box::new([bool_ty, bool_ty]), ()));
        let of_bools = fx.pure(&[bools], u8_ty);
        fx.overload("n1", "p", of_bools);
        (fx, wanted)
    };
    let templates = ["let r = p(({X}, 2u8));", "let r = p((2u8, {X}));"];
    assert_each_reaches(with_two_overloads_of_p, &every_diverging_form(&templates));
}

/// types §6: a method call reaches by its receiver, then its arguments in
/// order, and an argument of type `!` narrows nothing: with `m` over
/// `(i64, u8, u8)` and `(i64, bool, bool)`, `$n.m(2u8, X)` and
/// `$n.m(X, 2u8)` each reach the `(i64, u8, u8)` one, in every form.
#[test]
fn a_diverging_method_argument_narrows_no_overload_in_either_position() {
    let with_two_overloads_of_m = || {
        let mut fx = never_fixture();
        let (i64_ty, u8_ty, bool_ty) = (fx.int(IntTy::I64), fx.int(IntTy::U8), fx.ty(TyKind::Bool));
        fx.input("n", i64_ty);
        let of_bytes = fx.pure(&[i64_ty, u8_ty, u8_ty], u8_ty);
        let wanted = fx.overload("n0", "m", of_bytes);
        let of_bools = fx.pure(&[i64_ty, bool_ty, bool_ty], u8_ty);
        fx.overload("n1", "m", of_bools);
        (fx, wanted)
    };
    let templates = ["let r = $n.m(2u8, {X});", "let r = $n.m({X}, 2u8);"];
    assert_each_reaches(with_two_overloads_of_m, &every_diverging_form(&templates));
}

/// `never_fixture` with `step(i64) -> i64`, the one method of its name; and
/// `step`.
fn with_step() -> (Fixture, QualifiedRef) {
    let mut fx = never_fixture();
    let i64_ty = fx.int(IntTy::I64);
    let step = fx.pure(&[i64_ty], i64_ty);
    let wanted = fx.overload("n0", "step", step);
    (fx, wanted)
}

/// types §6: a method call's receiver is its reach's first position, and
/// `!` there says nothing of a type, so every candidate matches it: the
/// first adjustment, as written, is taken. `X.step()` of a diverging `X`
/// reaches `step(i64) -> i64`, the one method of its name, in every form.
#[test]
fn a_method_call_on_a_diverging_receiver_reaches_its_one_method_as_written() {
    assert_each_reaches(with_step, &every_diverging_form(&["let r = {X}.step();"]));
}

/// types §4-6: a diverging callee's type is what its use decides, and a
/// call decides a function type, whose flows, holds and calls nothing else
/// decides and take their defaults (types §5): `X(2)` of a diverging `X`,
/// as a condition, is admitted in every form.
#[test]
fn a_diverging_callee_is_decided_by_its_call() {
    assert_each_admitted(never_fixture, &every_diverging_form(&["if {X}(2) { 2 } else { 3 };"]));
}

/// types §4-6, patterns §1, §4: a pattern destructuring a diverging
/// scrutinee decides its type, and the names it binds are what their uses
/// decide: `if let`, `while let` and a tuple pattern of `match`, each
/// binding a condition's `bool`, are admitted in every form, `while let` in
/// every form but the loop's.
#[test]
fn a_pattern_over_a_diverging_scrutinee_binds_what_its_uses_decide() {
    let templates = [
        "let r = if let Some(u) = {X} { if u { 2 } else { 3 } } else { 4 };",
        "let r = match {X} { (u, w) => if u { 2 } else { 3 }, };",
    ];
    let mut sources = every_diverging_form(&templates);
    sources.extend(every_diverging_form_in_a_condition(&["while let Some(u) = {X} { if u { } }"]));
    assert_each_admitted(never_fixture, &sources);
}

/// types §3-6, operators §2-3: a diverging form entering a slot, an index's
/// `u64`, a let-bound lambda's parameter, a field stored into and a binding
/// assigned, and one lent to `core::add`, whose one impl takes `&i64`, on
/// either side, is admitted in every form.
#[test]
fn a_diverging_form_entering_a_slot_or_one_impls_parameter_is_admitted() {
    let templates = [
        "let v = [1, 2]; let e = v[{X}];",
        "let k = |z| -> if z { 1 } else { 2 }; let r = k({X});",
        "let o = { x: 1, }; o.x = {X};",
        "let z = 0; z = {X};",
        "let s = {X} + 1;",
        "let s = 1 + {X};",
    ];
    assert_each_admitted(never_fixture, &every_diverging_form(&templates));
}

/// types §4, §6: a declaration whose result is `!` gives no value, so it
/// matches every type its result is taken at. With `stop` over `() -> !`
/// and `() -> i64`, a `bool` condition leaves the `!` one alone, which is
/// reached; an `i64` leaves both, and the call is refused.
#[test]
fn a_declared_result_of_type_never_matches_every_type_the_result_is_taken_at() {
    let with_two_stops = || {
        let mut fx = never_fixture();
        let (never, i64_ty) = (fx.ty(TyKind::Never), fx.int(IntTy::I64));
        let stops = fx.pure(&[], never);
        let wanted = fx.overload("n0", "stop", stops);
        let gives = fx.pure(&[], i64_ty);
        fx.overload("n1", "stop", gives);
        (fx, wanted)
    };
    assert_each_reaches(with_two_stops, &["if stop() { 2 } else { 3 }".to_string()]);
    let (fx, _) = with_two_stops();
    let (checked, _) = fx.script("let x = stop(); let b = x == 1; 0");
    assert!(
        matches!(checked.refusals()[..], [RefusalKind::SeveralDeclarations { .. }]),
        "{:?}",
        checked.refusals()
    );
}
