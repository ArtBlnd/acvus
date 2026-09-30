//! A call site's segments: where each segment of a declared type
//! (`decl::segments`) lies among the positions of the concrete type a call
//! has.

use std::ops::Range;

use crate::decl::Declarations;
use crate::ty::{Concrete, Declared, FlowEnd, LifetimesOf, Poly, Ty, TyId, TyKind, Types, positions};

impl Declared {
    /// The declared type of the end `end` of this signature, in the
    /// declarations' table: the result, or the parameter at its index.
    ///
    /// # Panics
    /// The declaration, its impl or the end does not exist, or `end`
    /// is the callee's captures: a `Labelled` flow is read off an extern's
    /// signature, whose ends are its parameters and its result.
    pub fn end_ty(self, decls: &Declarations, end: FlowEnd) -> TyId<Poly> {
        let scheme = decls
            .scheme(self.function)
            .expect("a Labelled flow names a declared function");
        let signature = match self.r#impl {
            None => scheme.ty,
            Some(r#impl) => {
                scheme
                    .impls
                    .as_ref()
                    .and_then(|impls| impls.concrete.get(r#impl))
                    .expect("a Labelled flow names a concrete impl its declaration has")
                    .ty
            }
        };
        let TyKind::Fn { params, ret, .. } = decls.types().kind(signature) else {
            panic!("a declared function's signature is a function type")
        };
        match end {
            FlowEnd::Result => *ret,
            FlowEnd::Param(index) => *params
                .get(index)
                .expect("a Labelled flow's end is a parameter of its signature"),
            FlowEnd::Captures => panic!("a Labelled flow's ends are an extern's parameters and result"),
        }
    }
}

/// The positions of a value of `ty` each segment of `decl_ty` covers, in
/// the order `segments` gives them. `ty` instantiates `decl_ty`: a
/// type variable's part covers every position of what instantiates it.
///
/// # Panics
/// `ty` does not instantiate `decl_ty`. A call's concrete type is its
/// declaration's instantiated, so the two agree by construction.
pub fn segment_positions(
    poly: &Types<Poly>,
    decl_ty: TyId<Poly>,
    types: &Types<Concrete>,
    ty: Ty,
    decls: &Declarations,
) -> Vec<Range<usize>> {
    let mut walk = SegmentWalk {
        poly,
        types,
        decls,
        at: 0,
        out: Vec::new(),
    };
    walk.lay(decl_ty, ty);
    assert_eq!(
        walk.at,
        positions(types, decls, ty),
        "a declared type's segments lie within its instantiation's positions"
    );
    walk.out
}

struct SegmentWalk<'a> {
    poly: &'a Types<Poly>,
    types: &'a Types<Concrete>,
    decls: &'a Declarations,
    at: usize,
    out: Vec<Range<usize>>,
}

impl SegmentWalk<'_> {
    fn lay(&mut self, decl_ty: TyId<Poly>, ty: Ty) {
        acvus_utils::grow(|| self.lay_level(decl_ty, ty))
    }

    fn cover(&mut self, width: usize) {
        self.out.push(self.at..self.at + width);
        self.at += width;
    }

    fn lay_level(&mut self, decl_ty: TyId<Poly>, ty: Ty) {
        const DISAGREE: &str = "a call's concrete type instantiates its declared type";
        match (self.poly.kind(decl_ty), self.types.kind(ty)) {
            (TyKind::Var(_), _) => self.cover(positions(self.types, self.decls, ty)),
            (TyKind::Order | TyKind::Handle(_), _) => {
                panic!("a declared type holds no Handle and no Order")
            }
            (
                TyKind::Int(_)
                | TyKind::Float
                | TyKind::Char
                | TyKind::String
                | TyKind::Bool
                | TyKind::Unit
                | TyKind::Never
                | TyKind::Str,
                _,
            ) => {}
            (TyKind::Fn { .. }, TyKind::Fn { .. }) => self.cover(1),
            (TyKind::Ref(_, decl_target), TyKind::Ref(_, target)) => {
                self.cover(1);
                self.lay(decl_target.ty(), target.ty());
            }
            (
                TyKind::UserDefined {
                    id: decl_id,
                    type_args: decl_args,
                    ..
                },
                TyKind::UserDefined { id, type_args, .. },
            ) => {
                assert_eq!(decl_id, id, "{DISAGREE}");
                assert_eq!(decl_args.len(), type_args.len(), "{DISAGREE}");
                for _ in 0..self.decls.lifetimes_of(*id) {
                    self.cover(1);
                }
                for (decl_arg, arg) in decl_args.iter().zip(type_args.iter()) {
                    self.lay(decl_arg.ty(), arg.ty());
                }
            }
            (
                TyKind::Array(decl_element, _, _) | TyKind::Slice(decl_element),
                TyKind::Array(element, _, _) | TyKind::Slice(element),
            )
            | (TyKind::Option(decl_element, _), TyKind::Option(element, _)) => self.lay(*decl_element, *element),
            (TyKind::Result(decl_ok, decl_err, _), TyKind::Result(ok, err, _)) => {
                self.lay(*decl_ok, *ok);
                self.lay(*decl_err, *err);
            }
            (TyKind::Tuple(decl_items, _), TyKind::Tuple(items, _)) => {
                assert_eq!(decl_items.len(), items.len(), "{DISAGREE}");
                for (decl_item, item) in decl_items.iter().zip(items.iter()) {
                    self.lay(*decl_item, *item);
                }
            }
            (TyKind::Object(decl_object), TyKind::Object(object)) => {
                let (decl_fields, fields) = (decl_object.fields(), object.fields());
                assert_eq!(decl_fields.len(), fields.len(), "{DISAGREE}");
                for ((decl_name, decl_field), (name, field)) in decl_fields.iter().zip(fields.iter()) {
                    assert_eq!(decl_name, name, "{DISAGREE}");
                    self.lay(*decl_field, *field);
                }
            }
            (
                TyKind::Enum {
                    name: decl_name,
                    variants: decl_variants,
                    ..
                },
                TyKind::Enum { name, variants, .. },
            ) => {
                assert_eq!(decl_name, name, "{DISAGREE}");
                assert_eq!(decl_variants.len(), variants.len(), "{DISAGREE}");
                for ((decl_tag, decl_payload), (tag, payload)) in decl_variants.iter().zip(variants.iter()) {
                    assert_eq!(decl_tag, tag, "{DISAGREE}");
                    match (decl_payload, payload) {
                        (Some(decl_payload), Some(payload)) => self.lay(*decl_payload, *payload),
                        (None, None) => {}
                        _ => panic!("{DISAGREE}"),
                    }
                }
            }
            _ => panic!("{DISAGREE}"),
        }
    }
}
