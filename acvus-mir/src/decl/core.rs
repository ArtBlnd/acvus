//! `core`, declared inside the compiler (operators §2): the names of its
//! generic functions, the lang items the language's syntax lowers to, and
//! the tags of the primitive `Option` and `Result` (types §2). One place
//! for every reader: typeck, decl and the passes.
//!
//! The compiler builds each lang item's generic function from the table of
//! operators §2 (`Core::generic_functions`). A declaration set gives only
//! their impls.

use acvus_utils::{Astr, Interner, QualifiedRef};

use super::{DeclaredFacts, Scheme, TyVarBound};
use crate::ty::{Calls, CastTy, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Poly, TyId, TyKind, Types};

/// The names of `core`'s generic functions: the lang items (operators §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Core {
    pub add: QualifiedRef,
    pub sub: QualifiedRef,
    pub mul: QualifiedRef,
    pub div: QualifiedRef,
    pub rem: QualifiedRef,
    pub neg: QualifiedRef,
    pub eq: QualifiedRef,
    pub cmp: QualifiedRef,
    pub as_slice: QualifiedRef,
    pub as_slice_mut: QualifiedRef,
    pub as_str: QualifiedRef,
    pub display: QualifiedRef,
    pub clone: QualifiedRef,
    pub unwrap: QualifiedRef,
    pub default: QualifiedRef,
    /// `as_ref`, `as_mut`: the language's own lend of a transparent
    /// structured value's parts (operators §2), given no impls.
    pub as_ref: QualifiedRef,
    pub as_mut: QualifiedRef,
    pub tags: Tags,
    /// The names an `as` may cast to, each with its type (operators §5).
    cast_targets: [(Astr, CastTy); CAST_TARGETS],
    params: ParamNames,
}

const CAST_TARGETS: usize = IntTy::ALL.len() + 3;

/// The parameter names the lang items take: Rust's, and `out`, which
/// operators §2 names for `display`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParamNames {
    receiver: Astr,
    other: Astr,
    rhs: Astr,
    out: Astr,
}

/// The variants of the primitive `Option` and `Result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tags {
    pub some: Astr,
    pub none: Astr,
    pub ok: Astr,
    pub err: Astr,
}

impl Core {
    pub fn of(interner: &Interner) -> Core {
        let namespace = interner.namespace(&[interner.intern("core")]);
        let name = |name: &str| QualifiedRef::qualified(namespace, interner.intern(name));
        Core {
            add: name("add"),
            sub: name("sub"),
            mul: name("mul"),
            div: name("div"),
            rem: name("rem"),
            neg: name("neg"),
            eq: name("eq"),
            cmp: name("cmp"),
            as_slice: name("as_slice"),
            as_slice_mut: name("as_slice_mut"),
            as_str: name("as_str"),
            display: name("display"),
            clone: name("clone"),
            unwrap: name("unwrap"),
            default: name("default"),
            as_ref: name("as_ref"),
            as_mut: name("as_mut"),
            tags: Tags {
                some: interner.intern("Some"),
                none: interner.intern("None"),
                ok: interner.intern("Ok"),
                err: interner.intern("Err"),
            },
            cast_targets: std::array::from_fn(|at| {
                let target = match IntTy::ALL.get(at) {
                    Some(width) => CastTy::Int(*width),
                    None => [CastTy::F64, CastTy::Bool, CastTy::Char][at - IntTy::ALL.len()],
                };
                (interner.intern(target.name()), target)
            }),
            params: ParamNames {
                receiver: interner.intern("self"),
                other: interner.intern("other"),
                rhs: interner.intern("rhs"),
                out: interner.intern("out"),
            },
        }
    }

    /// Every lang item's generic function, as the table of operators §2
    /// states it: its type, each type variable bounded by any type, and its
    /// parameters' names. The table states no effect, and a lang item means
    /// something only through its impls (operators §2), so each one's
    /// effect is a variable its impls choose. Its flows are what `verify`
    /// writes for a signature stating no lifetimes (extern-fn §4).
    pub(super) fn generic_functions(&self, types: &mut Types<Poly>) -> Vec<(QualifiedRef, Scheme)> {
        let mut table = Table { types, names: self.params };
        let arithmetic = [self.add, self.sub, self.mul, self.div, self.rem];
        let mut functions: Vec<(QualifiedRef, Scheme)> = arithmetic
            .into_iter()
            .map(|item| (item, table.binary_to_chosen()))
            .collect();
        functions.extend([
            (self.clone, table.clone()),
            (self.eq, table.compare(TyKind::Bool)),
            (self.cmp, table.compare(TyKind::Int(IntTy::I64))),
            (self.neg, table.neg()),
            (self.display, table.display()),
            (self.as_slice, table.view_of_slice(Mutability::Shared)),
            (self.as_slice_mut, table.view_of_slice(Mutability::Mut)),
            (self.as_str, table.view_of_str()),
            (self.unwrap, table.unwrap()),
            (self.default, table.default()),
            (self.as_ref, table.lend(Mutability::Shared)),
            (self.as_mut, table.lend(Mutability::Mut)),
        ]);
        functions
    }

    /// The name a lang item gives its receiver, which a function the
    /// compiler makes of one parameter gives its one parameter too.
    pub fn receiver_param(&self) -> Astr {
        self.params.receiver
    }

    /// The type an `as` names by `name`, if it names one (operators §5).
    pub fn cast_target(&self, name: Astr) -> Option<CastTy> {
        self.cast_targets
            .iter()
            .find(|(written, _)| *written == name)
            .map(|(_, target)| *target)
    }

    /// Whether `function` is a lang item's name.
    pub(super) fn names(&self, function: QualifiedRef) -> bool {
        [
            self.add,
            self.sub,
            self.mul,
            self.div,
            self.rem,
            self.neg,
            self.eq,
            self.cmp,
            self.as_slice,
            self.as_slice_mut,
            self.as_str,
            self.display,
            self.clone,
            self.unwrap,
            self.default,
            self.as_ref,
            self.as_mut,
        ]
        .contains(&function)
    }

    /// Whether `function` is a lang item whose meaning is the language's
    /// own, which no declaration gives impls (operators §2).
    pub(super) fn takes_no_impls(&self, function: QualifiedRef) -> bool {
        [self.as_ref, self.as_mut].contains(&function)
    }

    /// The mutability `function` lends a structure's parts at, where it is
    /// `as_ref` or `as_mut` (operators §2).
    pub fn lends(&self, function: QualifiedRef) -> Option<Mutability> {
        match function {
            _ if function == self.as_ref => Some(Mutability::Shared),
            _ if function == self.as_mut => Some(Mutability::Mut),
            _ => None,
        }
    }
}

/// The table of operators §2, written into one `Types<Poly>`.
struct Table<'t> {
    types: &'t mut Types<Poly>,
    names: ParamNames,
}

impl Table<'_> {
    fn var(&mut self, at: u32) -> TyId<Poly> {
        self.types.intern(TyKind::Var(at))
    }

    fn reference(&mut self, mutability: Mutability, target: TyId<Poly>) -> TyId<Poly> {
        self.types.intern(TyKind::Ref(mutability, target))
    }

    fn scheme(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>, names: &[Astr], variables: usize) -> Scheme {
        let ty = self.types.intern(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Var(0),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        Scheme {
            ty,
            param_names: names.into(),
            bounds: vec![TyVarBound::Any; variables].into(),
            impls: None,
            requires: Box::new([]),
            facts: DeclaredFacts::default(),
            runs: Box::new([]),
            specialized: Box::new([]),
        }
    }

    /// `add`, `sub`, `mul`, `div`, `rem`: `<T, O>(&T, &T) -> O`.
    fn binary_to_chosen(&mut self) -> Scheme {
        let (t, o) = (self.var(0), self.var(1));
        let shared_t = self.reference(Mutability::Shared, t);
        let names = [self.names.receiver, self.names.rhs];
        self.scheme(&[shared_t, shared_t], o, &names, 2)
    }

    /// `clone<T>(&T) -> T`.
    fn clone(&mut self) -> Scheme {
        let t = self.var(0);
        let shared_t = self.reference(Mutability::Shared, t);
        let names = [self.names.receiver];
        self.scheme(&[shared_t], t, &names, 1)
    }

    /// `eq<T>(&T, &T) -> bool`, `cmp<T>(&T, &T) -> i64`.
    fn compare(&mut self, answer: TyKind<Poly>) -> Scheme {
        let t = self.var(0);
        let shared_t = self.reference(Mutability::Shared, t);
        let answer = self.types.intern(answer);
        let names = [self.names.receiver, self.names.other];
        self.scheme(&[shared_t, shared_t], answer, &names, 1)
    }

    /// `neg<T, O>(&T) -> O`.
    fn neg(&mut self) -> Scheme {
        let (t, o) = (self.var(0), self.var(1));
        let shared_t = self.reference(Mutability::Shared, t);
        let names = [self.names.receiver];
        self.scheme(&[shared_t], o, &names, 2)
    }

    /// `display<T>(&T, out: &mut String)`.
    fn display(&mut self) -> Scheme {
        let t = self.var(0);
        let shared_t = self.reference(Mutability::Shared, t);
        let string = self.types.intern(TyKind::String);
        let out = self.reference(Mutability::Mut, string);
        let unit = self.types.intern(TyKind::Unit);
        let names = [self.names.receiver, self.names.out];
        self.scheme(&[shared_t, out], unit, &names, 1)
    }

    /// `as_slice<C, T>(&C) -> &[T]`, `as_slice_mut<C, T>(&mut C) -> &mut [T]`.
    fn view_of_slice(&mut self, mutability: Mutability) -> Scheme {
        let (c, t) = (self.var(0), self.var(1));
        let lent_c = self.reference(mutability, c);
        let slice = self.types.intern(TyKind::Slice(t));
        let lent_slice = self.reference(mutability, slice);
        let names = [self.names.receiver];
        self.scheme(&[lent_c], lent_slice, &names, 2)
    }

    /// `as_str<S>(&S) -> &str`.
    fn view_of_str(&mut self) -> Scheme {
        let s = self.var(0);
        let shared_s = self.reference(Mutability::Shared, s);
        let str_ty = self.types.intern(TyKind::Str);
        let shared_str = self.reference(Mutability::Shared, str_ty);
        let names = [self.names.receiver];
        self.scheme(&[shared_s], shared_str, &names, 1)
    }

    /// `unwrap<W, T>(W) -> T`.
    fn unwrap(&mut self) -> Scheme {
        let (w, t) = (self.var(0), self.var(1));
        let names = [self.names.receiver];
        self.scheme(&[w], t, &names, 2)
    }

    /// `as_ref<S, T>(&S) -> T`, `as_mut<S, T>(&mut S) -> T`: `T` is `S`
    /// with each part lent, which no signature states, so the language
    /// answers it at the call (operators §2).
    fn lend(&mut self, mutability: Mutability) -> Scheme {
        let (s, t) = (self.var(0), self.var(1));
        let lent_s = self.reference(mutability, s);
        let names = [self.names.receiver];
        self.scheme(&[lent_s], t, &names, 2)
    }

    /// `default<T>() -> T`.
    fn default(&mut self) -> Scheme {
        let t = self.var(0);
        self.scheme(&[], t, &[], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lang item's type written as operators §2's table writes it, over
    /// the type variables `T` = 0 and `O`, `C`, `W` = 1 as the table names
    /// them in order.
    struct Expected {
        params: Vec<TyKind<Poly>>,
        ret: TyKind<Poly>,
        names: Vec<&'static str>,
        variables: usize,
    }

    fn check(types: &Types<Poly>, interner: &Interner, scheme: &Scheme, expected: Expected) {
        let TyKind::Fn {
            params,
            ret,
            effect,
            calls,
            holds,
            ..
        } = types.kind(scheme.ty)
        else {
            panic!("a lang item is a function");
        };
        let params: Vec<TyKind<Poly>> = params.iter().map(|param| types.kind(*param).clone()).collect();
        assert_eq!(params, expected.params);
        assert_eq!(types.kind(*ret), &expected.ret);
        assert_eq!(effect, &EffectTerm::Var(0), "the impls choose the effect");
        assert_eq!((*calls, *holds), (Calls::Many, Holds::Nothing));
        let names: Vec<Astr> = expected.names.iter().map(|name| interner.intern(name)).collect();
        assert_eq!(&scheme.param_names[..], &names[..]);
        assert_eq!(&scheme.bounds[..], &vec![TyVarBound::Any; expected.variables][..]);
        assert!(scheme.impls.is_none() && scheme.requires.is_empty());
    }

    /// operators §5: `as` names the numbers, `bool` and `char`, and nothing
    /// else.
    #[test]
    fn the_cast_targets_are_the_numbers_bool_and_char() {
        let interner = Interner::new();
        let core = Core::of(&interner);
        for name in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f64", "bool", "char"] {
            let target = core.cast_target(interner.intern(name)).expect("a cast target");
            assert_eq!(target.name(), name);
        }
        assert_eq!(core.cast_target(interner.intern("String")), None);
    }

    /// operators §2: every lang item's generic function is the table's.
    #[test]
    fn every_lang_item_is_the_tables() {
        let interner = Interner::new();
        let core = Core::of(&interner);
        let mut types = Types::new();
        let functions = core.generic_functions(&mut types);
        let mut at = |kind: TyKind<Poly>| types.intern(kind);
        let (t, second) = (at(TyKind::Var(0)), at(TyKind::Var(1)));
        let reference = |mutability, target| TyKind::Ref(mutability, target);
        let shared = |target| reference(Mutability::Shared, target);
        let string = at(TyKind::String);
        let slice = at(TyKind::Slice(second));
        let str_ty = at(TyKind::Str);
        let binary = |ret| Expected {
            params: vec![shared(t), shared(t)],
            ret,
            names: vec!["self", "rhs"],
            variables: 2,
        };
        let mut expected: Vec<(QualifiedRef, Expected)> = [core.add, core.sub, core.mul, core.div, core.rem]
            .into_iter()
            .map(|item| (item, binary(TyKind::Var(1))))
            .collect();
        expected.extend([
            (
                core.clone,
                Expected {
                    params: vec![shared(t)],
                    ret: TyKind::Var(0),
                    names: vec!["self"],
                    variables: 1,
                },
            ),
            (
                core.eq,
                Expected {
                    params: vec![shared(t), shared(t)],
                    ret: TyKind::Bool,
                    names: vec!["self", "other"],
                    variables: 1,
                },
            ),
            (
                core.cmp,
                Expected {
                    params: vec![shared(t), shared(t)],
                    ret: TyKind::Int(IntTy::I64),
                    names: vec!["self", "other"],
                    variables: 1,
                },
            ),
            (
                core.neg,
                Expected {
                    params: vec![shared(t)],
                    ret: TyKind::Var(1),
                    names: vec!["self"],
                    variables: 2,
                },
            ),
            (
                core.display,
                Expected {
                    params: vec![shared(t), reference(Mutability::Mut, string)],
                    ret: TyKind::Unit,
                    names: vec!["self", "out"],
                    variables: 1,
                },
            ),
            (
                core.as_slice,
                Expected {
                    params: vec![shared(t)],
                    ret: shared(slice),
                    names: vec!["self"],
                    variables: 2,
                },
            ),
            (
                core.as_slice_mut,
                Expected {
                    params: vec![reference(Mutability::Mut, t)],
                    ret: reference(Mutability::Mut, slice),
                    names: vec!["self"],
                    variables: 2,
                },
            ),
            (
                core.as_str,
                Expected {
                    params: vec![shared(t)],
                    ret: shared(str_ty),
                    names: vec!["self"],
                    variables: 1,
                },
            ),
            (
                core.unwrap,
                Expected {
                    params: vec![TyKind::Var(0)],
                    ret: TyKind::Var(1),
                    names: vec!["self"],
                    variables: 2,
                },
            ),
            (
                core.default,
                Expected {
                    params: vec![],
                    ret: TyKind::Var(0),
                    names: vec![],
                    variables: 1,
                },
            ),
            (
                core.as_ref,
                Expected {
                    params: vec![shared(t)],
                    ret: TyKind::Var(1),
                    names: vec!["self"],
                    variables: 2,
                },
            ),
            (
                core.as_mut,
                Expected {
                    params: vec![reference(Mutability::Mut, t)],
                    ret: TyKind::Var(1),
                    names: vec!["self"],
                    variables: 2,
                },
            ),
        ]);
        assert_eq!(functions.len(), expected.len(), "one generic function per lang item");
        for (item, expected) in expected {
            let (_, scheme) = functions
                .iter()
                .find(|(id, _)| *id == item)
                .expect("every lang item is built");
            assert!(core.names(item));
            check(&types, &interner, scheme, expected);
        }
    }
}
