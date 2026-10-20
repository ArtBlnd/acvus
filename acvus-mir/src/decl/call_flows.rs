//! What a call of a declared function moves between the positions of its
//! arguments and its result (ownership §5, extern-fn §4), read off the
//! declaration's lifetimes at the call's concrete types. An extern fn is
//! only called, never a value (extern-fn §1), so its lifetimes are read
//! here, at its calls, and never carried in a function type.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::ty::{Concrete, Declared, FlowEnd, LifetimesOf, Poly, Ty, TyId, TyKind, Types, positions};

use super::labels::SegmentLabel;
use super::segments::{Segment, segments};
use super::verify::{FittedEnd, fitted_ends, outlives_closure};
use super::Declarations;

/// For each output end of a call (its result, and each parameter through
/// what the call may write into), and for each position of that end, the
/// input positions it takes, each as (parameter, position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallFlows {
    pub into: Vec<(FlowEnd, Box<[Box<[(usize, usize)]>]>)>,
}

/// The flows of a call of `declared` at its concrete parameter and result
/// types: `None` where the declaration states no lifetimes, which is every
/// flow (extern-fn §4).
///
/// Each written segment of an output end takes every input segment of its
/// lifetime, of a lifetime that outlives it, or of its type variable. Two
/// segments of one type variable meet position by position; any other
/// pair, every position of the output segment from every position of the
/// input segment. An output end that takes nothing is left out.
///
/// # Panics
/// `declared` names no signature of a declaration, its lifetimes do not
/// fit its type, or `params` and `ret` do not instantiate that type.
/// `verify` refuses lifetimes that do not fit (`LabelsDoNotFit`), and a
/// call's concrete types are its declaration's instantiated.
pub fn call_flows(decls: &Declarations, declared: Declared, types: &Types<Concrete>, params: &[Ty], ret: Ty) -> Option<CallFlows> {
    let scheme = decls
        .scheme(declared.function)
        .unwrap_or_else(|| panic!("a call names {:?}, which is declared", declared.function));
    let labelled = scheme.facts.lifetimes.as_ref()?;
    let TyKind::Fn { params: declared_params, ret: declared_ret, .. } = decls.types().kind(scheme.ty) else {
        panic!("a declared signature with lifetimes is a function type: verify refuses one that is not (LabelsDoNotFit)")
    };
    assert_eq!(declared_params.len(), params.len(), "a call passes one argument per parameter of its declaration");
    const FIT: &str = "a declaration's lifetimes fit its type: verify refuses those that do not (LabelsDoNotFit)";
    let Ok(ends) = fitted_ends(decls, declared_params, *declared_ret, labelled) else { panic!("{FIT}") };
    let Ok(outlives) = outlives_closure(labelled, &ends) else { panic!("{FIT}") };
    let reaches = |from: SegmentLabel, to: SegmentLabel| match (from, to) {
        (SegmentLabel::Lifetime(longer), SegmentLabel::Lifetime(shorter)) => {
            longer == shorter || outlives.contains(&(longer, shorter))
        }
        (SegmentLabel::TypeVar(from), SegmentLabel::TypeVar(to)) => from == to,
        (SegmentLabel::Lifetime(_), SegmentLabel::TypeVar(_)) | (SegmentLabel::TypeVar(_), SegmentLabel::Lifetime(_)) => false,
    };
    let laid = |end: &FittedEnd<'_>| {
        let (declared_ty, ty) = match end.end {
            FlowEnd::Result => (*declared_ret, ret),
            FlowEnd::Param(at) => (declared_params[at], params[at]),
            FlowEnd::Captures => panic!("a declared signature's ends are its parameters and its result"),
        };
        let laid = LaidEnd {
            segments: segments(decls.types(), decls, declared_ty),
            ranges: segment_positions(decls.types(), declared_ty, types, ty, decls),
            width: positions(types, decls, ty),
        };
        assert_eq!(laid.segments.len(), laid.ranges.len(), "one range per segment of the declared type");
        laid
    };
    let inputs: Vec<InputEnd<'_>> = ends
        .iter()
        .filter_map(|end| match end.end {
            FlowEnd::Param(param) => Some(InputEnd { param, labels: end.labels, laid: laid(end) }),
            FlowEnd::Result | FlowEnd::Captures => None,
        })
        .collect();

    let mut into = Vec::new();
    for to in &ends {
        let out = laid(to);
        let mut takes: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); out.width];
        for (at, out_label) in to.labels.iter().enumerate().filter(|(at, _)| to.written[*at]) {
            let out_range = out.ranges[at].clone();
            for input in &inputs {
                for (from, in_label) in input.labels.iter().enumerate() {
                    if !reaches(*in_label, *out_label) {
                        continue;
                    }
                    let in_range = input.laid.ranges[from].clone();
                    let aligned = out.segments[at].aligned_with(input.laid.segments[from]);
                    for (k, position) in out_range.clone().enumerate() {
                        match aligned {
                            true => {
                                takes[position].insert((input.param, in_range.start + k));
                            }
                            false => takes[position].extend(in_range.clone().map(|from| (input.param, from))),
                        }
                    }
                }
            }
        }
        if takes.iter().any(|taken| !taken.is_empty()) {
            into.push((to.end, takes.into_iter().map(|taken| taken.into_iter().collect()).collect()));
        }
    }
    Some(CallFlows { into })
}

/// One end of a call: its declared type's segments, the positions of its
/// concrete type each covers, and how many positions it has.
struct LaidEnd {
    segments: Vec<Segment>,
    ranges: Vec<Range<usize>>,
    width: usize,
}

/// A parameter of a call as an input: its labels, one per segment, and
/// where those segments lie.
struct InputEnd<'s> {
    param: usize,
    labels: &'s [SegmentLabel],
    laid: LaidEnd,
}

/// The positions of a value of `ty` each segment of `decl_ty` covers, in
/// the order `segments` gives them. `ty` instantiates `decl_ty`: a
/// type variable's part covers every position of what instantiates it.
///
/// # Panics
/// `ty` does not instantiate `decl_ty`. A call's concrete type is its
/// declaration's instantiated, so the two agree by construction.
fn segment_positions(
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
            (TyKind::Order | TyKind::Handle(_) | TyKind::Closures(_), _) => {
                panic!("a declared type holds no Handle, Order or Closures")
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
                self.lay(*decl_target, *target);
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
                    self.lay(*decl_arg, *arg);
                }
            }
            (
                TyKind::Array(decl_element, _) | TyKind::Slice(decl_element),
                TyKind::Array(element, _) | TyKind::Slice(element),
            )
            | (TyKind::Option(decl_element), TyKind::Option(element)) => self.lay(*decl_element, *element),
            (TyKind::Result(decl_ok, decl_err), TyKind::Result(ok, err)) => {
                self.lay(*decl_ok, *ok);
                self.lay(*decl_err, *err);
            }
            (TyKind::Tuple(decl_items), TyKind::Tuple(items)) => {
                assert_eq!(decl_items.len(), items.len(), "{DISAGREE}");
                for (decl_item, item) in decl_items.iter().zip(items.iter()) {
                    self.lay(*decl_item, *item);
                }
            }
            (TyKind::Object { fields: decl_fields, .. }, TyKind::Object { fields, .. }) => {
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
