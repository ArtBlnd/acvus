//! Effect and identity forms against the compiler. Declarations are built
//! by hand as a registry would give them, programs are checked by typeck,
//! and each outcome is printed beside its expectation. An effect is
//! observed through a pure slot: `apply` takes a pure function, so a lambda
//! whose work the compiler finds opaque is refused there.

use std::fmt::Debug;

use acvus_ast::parse_script;
use acvus_mir::decl::{
    Core, Declarations, DeclaredFacts, GenericFunction, ImplKind, ImplOf, LoopMarkers, RawDeclarations, Requirement, Scheme,
    TyVarBound, UserTypeDecl, declared_fn_flows,
};
use acvus_mir::ty::{Calls, Concrete, Effect, EffectTerm, FlowTerm, Holds, IdentityTerm, IntTy, Mark, MarkTerm, Poly, TyId, TyKind, Types};
use acvus_mir::typeck::{Inputs, Interfaces, check_script};
use acvus_utils::{Interner, QualifiedRef};

struct Fx {
    names: Interner,
    raw: RawDeclarations,
}

impl Fx {
    fn new() -> Fx {
        let names = Interner::new();
        let core = Core::of(&names);
        let raw = RawDeclarations {
            types: Types::new(),
            functions: Vec::new(),
            generic_functions: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core,
        };
        Fx { names, raw }
    }

    fn id(&self, name: &str) -> QualifiedRef {
        QualifiedRef::root(self.names.intern(name))
    }

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    fn var(&mut self, at: u32) -> TyId<Poly> {
        self.ty(TyKind::Var(at))
    }

    fn func(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, effect: EffectTerm<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect,
            flows: FlowTerm::Known(declared_fn_flows()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    }

    fn user(&mut self, name: &str, params: usize, effect: bool, identity: bool) -> QualifiedRef {
        let id = self.id(name);
        self.raw.user_types.push(UserTypeDecl {
            id,
            type_params: vec![TyVarBound::Any; params].into(),
            carries_effect: effect,
            carries_identity: identity,
            lifetimes: 0,
        });
        id
    }

    fn named(&mut self, id: QualifiedRef, args: &[TyId<Poly>], effect: Option<EffectTerm<Poly>>, identity: Option<IdentityTerm<Poly>>) -> TyId<Poly> {
        self.ty(TyKind::UserDefined { id, type_args: args.into(), effect, identity, marked: MarkTerm::Known(Mark::Unmarked) })
    }

    fn scheme(&self, ty: TyId<Poly>, type_vars: usize, runs: &[u32]) -> Scheme {
        let arity = match self.raw.types.kind(ty) {
            TyKind::Fn { params, .. } => params.len(),
            _ => 0,
        };
        Scheme {
            ty,
            param_names: (0..arity).map(|at| self.names.intern(&format!("p{at}"))).collect(),
            bounds: vec![TyVarBound::Settled; type_vars].into(),
            impl_of: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: runs.into(),
            awaits: false,
            specialized: Box::new([]),
        }
    }

    fn declare(&mut self, name: &str, scheme: Scheme) -> QualifiedRef {
        let id = self.id(name);
        self.raw.functions.push((id, scheme));
        id
    }

    /// `apply(f: fn(i64) -> i64 pure) -> i64` and `touch: fn(i64) -> i64`, opaque.
    fn slot(&mut self) {
        let int = self.int();
        let pure = self.func(&[int], int, EffectTerm::Known(Effect::pure()));
        let apply = self.func(&[pure], int, EffectTerm::Known(Effect::pure()));
        let scheme = self.scheme(apply, 0, &[]);
        self.declare("apply", scheme);
        let touch = self.func(&[int], int, EffectTerm::Known(Effect::opaque()));
        let scheme = self.scheme(touch, 0, &[]);
        self.declare("touch", scheme);
    }
}

fn short(kind: &impl Debug) -> String {
    let text = format!("{kind:?}");
    let end = text.find(['(', ' ', '{']).unwrap_or(text.len());
    text[..end].to_string()
}

fn verified(fx: Fx) -> String {
    match Declarations::verify(fx.raw) {
        Ok(_) => "verify admitted".to_string(),
        Err(refused) => format!("verify refused {:?}", refused.iter().map(short).collect::<Vec<_>>()),
    }
}

fn checked(fx: Fx, source: &str) -> String {
    let decls = match Declarations::verify(fx.raw) {
        Ok(decls) => decls,
        Err(refused) => return format!("verify refused {:?}", refused.iter().map(short).collect::<Vec<_>>()),
    };
    let script = parse_script(&fx.names, source).expect("the probe's source parses");
    let mut types: Types<Concrete> = Types::new();
    let typeck = check_script(&fx.names, &decls, &Interfaces::new(), Inputs::Open, &mut types, &script);
    match typeck.refusals.is_empty() {
        true => "admitted".to_string(),
        false => format!("refused {:?}", typeck.refusals.iter().map(|refusal| short(&refusal.kind)).collect::<Vec<_>>()),
    }
}

fn report(name: &str, expect: &str, got: String) {
    let verdict = match got.starts_with(expect) {
        true => "as-expected",
        false => "UNEXPECTED",
    };
    println!("{name:<44} expect={expect:<16} got={got}  {verdict}");
}

/// `Sealed with E`; `seal(fn with E) -> Sealed with E`, which carries E;
/// `run(Sealed with E)`, which runs E; `apply2` and `seal2`, two
/// functions of one effect variable.
fn carrying() -> Fx {
    let mut fx = Fx::new();
    fx.slot();
    let int = fx.int();
    let sealed_id = fx.user("Sealed", 0, true, false);
    let sealed = fx.named(sealed_id, &[], Some(EffectTerm::Var(0)), None);
    let work = fx.func(&[int], int, EffectTerm::Var(0));
    let pure = EffectTerm::Known(Effect::pure());
    let seal = fx.func(&[work], sealed, pure.clone());
    let scheme = fx.scheme(seal, 0, &[]);
    fx.declare("seal", scheme);
    let seal2 = fx.func(&[work, work], sealed, pure.clone());
    let scheme = fx.scheme(seal2, 0, &[]);
    fx.declare("seal2", scheme);
    let run = fx.func(&[sealed], int, pure.clone());
    let scheme = fx.scheme(run, 0, &[0]);
    fx.declare("run", scheme);
    let apply2 = fx.func(&[work, work], int, pure);
    let scheme = fx.scheme(apply2, 0, &[0]);
    fx.declare("apply2", scheme);
    fx
}

/// One declaration `consume(Sealed with E) -> i64` written one way, beside
/// `Sealed`.
fn consuming(own: EffectTerm<Poly>, runs: &[u32]) -> Fx {
    let mut fx = Fx::new();
    let int = fx.int();
    let sealed_id = fx.user("Sealed", 0, true, false);
    let sealed = fx.named(sealed_id, &[], Some(EffectTerm::Var(0)), None);
    let consume = fx.func(&[sealed], int, own);
    let scheme = fx.scheme(consume, 0, runs);
    fx.declare("consume", scheme);
    fx
}

/// `Noisy`, made by `noisy()`; the generic function `step(S) -> i64`,
/// its effect a type parameter of its signature; its impl at `Noisy`,
/// opaque; `drive(S) -> i64`, which requires `step` at `S` and states the
/// requirement's effect as `pattern_effect`, running `runs`.
fn requiring(pattern_effect: EffectTerm<Poly>, runs: &[u32]) -> Fx {
    requiring_with(EffectTerm::Known(Effect::pure()), pattern_effect, runs)
}

/// As `requiring`, `drive`'s own function type carrying `own`.
fn requiring_with(own: EffectTerm<Poly>, pattern_effect: EffectTerm<Poly>, runs: &[u32]) -> Fx {
    let mut fx = Fx::new();
    fx.slot();
    let int = fx.int();
    let noisy_id = fx.user("Noisy", 0, false, false);
    let noisy = fx.named(noisy_id, &[], None, None);
    let make = fx.func(&[], noisy, EffectTerm::Known(Effect::pure()));
    let scheme = fx.scheme(make, 0, &[]);
    fx.declare("noisy", scheme);
    let s = fx.var(0);
    let generic_ty = fx.func(&[s], int, EffectTerm::Var(0));
    let step = fx.id("step");
    let names = (0..1).map(|at| fx.names.intern(&format!("p{at}"))).collect();
    fx.raw.generic_functions.push((
        step,
        GenericFunction { ty: generic_ty, param_names: names, bounds: vec![TyVarBound::Settled].into(), specialized: Box::new([]) },
    ));
    let impl_ty = fx.func(&[noisy], int, EffectTerm::Known(Effect::opaque()));
    let mut scheme = fx.scheme(impl_ty, 0, &[]);
    scheme.impl_of = Some(ImplOf { generic_function: step, kind: ImplKind::Concrete });
    fx.declare("step#0", scheme);
    let pattern = fx.func(&[s], int, pattern_effect);
    let drive_ty = fx.func(&[s], int, own);
    let mut scheme = fx.scheme(drive_ty, 1, runs);
    scheme.requires = vec![Requirement { generic_function: step, pattern, fact: None }].into();
    fx.declare("drive", scheme);
    fx
}

/// `Str with I`, made fresh by `make()`; `dup(Str with I) -> Str with I`;
/// `same(Str with I, Str with I) -> i64`; `Joined<A, B>` by `join(A, B)`,
/// taken apart by `first` and `second`.
fn identities() -> Fx {
    let mut fx = Fx::new();
    let int = fx.int();
    let str_id = fx.user("Str", 0, false, true);
    let str_i = fx.named(str_id, &[], None, Some(IdentityTerm::Var(0)));
    let pure = EffectTerm::Known(Effect::pure());
    let make = fx.func(&[], str_i, pure.clone());
    let scheme = fx.scheme(make, 0, &[]);
    fx.declare("make", scheme);
    let dup = fx.func(&[str_i], str_i, pure.clone());
    let scheme = fx.scheme(dup, 0, &[]);
    fx.declare("dup", scheme);
    let same = fx.func(&[str_i, str_i], int, pure.clone());
    let scheme = fx.scheme(same, 0, &[]);
    fx.declare("same", scheme);
    let joined_id = fx.user("Joined", 2, false, false);
    let (a, b) = (fx.var(0), fx.var(1));
    let joined = fx.named(joined_id, &[a, b], None, None);
    let join = fx.func(&[a, b], joined, pure.clone());
    let scheme = fx.scheme(join, 2, &[]);
    fx.declare("join", scheme);
    let first = fx.func(&[joined], a, pure.clone());
    let scheme = fx.scheme(first, 2, &[]);
    fx.declare("first", scheme);
    let second = fx.func(&[joined], b, pure);
    let scheme = fx.scheme(second, 2, &[]);
    fx.declare("second", scheme);
    fx
}

fn main() {
    println!("-- effects carried and consumed");
    report("e01 opaque work in the pure slot", "refused", checked(carrying(), "apply(|x| -> touch(x))"));
    report("e02 carried, then run", "refused", checked(carrying(), "apply(|x| -> run(seal(|y| -> touch(y))))"));
    report("e03 carried pure work, then run", "admitted", checked(carrying(), "apply(|x| -> run(seal(|y| -> y)))"));
    report("e04 one variable, pure and opaque, run", "refused", checked(carrying(), "apply(|x| -> apply2(|a| -> a, |b| -> touch(b)))"));
    report("e05 one variable, both pure, run", "admitted", checked(carrying(), "apply(|x| -> apply2(|a| -> a, |b| -> b))"));
    report("e06 one variable, mixed, carried, run", "refused", checked(carrying(), "apply(|x| -> run(seal2(|a| -> a, |b| -> touch(b))))"));
    report("e07 one variable, mixed, outside a slot", "admitted", checked(carrying(), "apply2(|a| -> a, |b| -> touch(b))"));

    println!("-- how a declaration consumes its effect");
    report("c01 own effect E, runs nothing", "verify refused", verified(consuming(EffectTerm::Var(0), &[])));
    report("c02 own effect pure, runs E", "verify admitted", verified(consuming(EffectTerm::Known(Effect::pure()), &[0])));
    report("c03 neither consumed nor carried", "verify refused", verified(consuming(EffectTerm::Known(Effect::pure()), &[])));

    println!("-- an impl's effect, reached through a requirement");
    report("r01 the generic function called directly", "refused", checked(requiring(EffectTerm::Var(0), &[]), "apply(|x| -> step(noisy()))"));
    report("r02 through drive, pattern effect E", "refused", checked(requiring(EffectTerm::Var(0), &[]), "apply(|x| -> drive(noisy()))"));
    report("r03 through drive, pattern E, runs E", "refused", checked(requiring(EffectTerm::Var(0), &[0]), "apply(|x| -> drive(noisy()))"));
    report("r04 drive outside a slot", "admitted", checked(requiring(EffectTerm::Var(0), &[]), "drive(noisy())"));
    report("r05 drive with E, its requirement with E", "refused", checked(requiring_with(EffectTerm::Var(0), EffectTerm::Var(0), &[]), "apply(|x| -> drive(noisy()))"));
    report("r06 the same, running E", "refused", checked(requiring_with(EffectTerm::Var(0), EffectTerm::Var(0), &[0]), "apply(|x| -> drive(noisy()))"));
    report("r07 drive with E outside a slot", "admitted", checked(requiring_with(EffectTerm::Var(0), EffectTerm::Var(0), &[]), "drive(noisy())"));

    println!("-- identities");
    report("i01 one identity given to its result", "admitted", checked(identities(), "let a = make(); same(dup(a), a)"));
    report("i02 two identities made one", "refused", checked(identities(), "let a = make(); let b = make(); same(a, b)"));
    report("i03 carried by a type argument, kept", "admitted", checked(identities(), "let a = make(); let b = make(); same(first(join(a, b)), a)"));
    report("i04 carried by a type argument, swapped", "refused", checked(identities(), "let a = make(); let b = make(); same(first(join(a, b)), b)"));
    report("i05 the other argument, kept", "admitted", checked(identities(), "let a = make(); let b = make(); same(second(join(a, b)), b)"));
}
