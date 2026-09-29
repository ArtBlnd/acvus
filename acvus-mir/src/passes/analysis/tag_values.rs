//! Which variants each value may hold, and each part of a slot's content,
//! within the one body (matchck, patterns §3): sparse, over the body's
//! def-use and the stores into its slots, and not over its points. A
//! variant the body makes holds its tag; a block's parameter, and a slot's
//! content, hold the join of what enters them; a part is what the value it
//! was made of holds at that part. A value from outside the body, a
//! parameter, a capture, a call's result or a context or input read, holds
//! `Any`, and so does a slot a `&mut` lends, since what is stored through
//! the reference is not seen by name.
//!
//! The facts are the body's def-use summary; `at` and `in_slot` read it on
//! demand. A cycle through parameters and slots carries no tag of its own,
//! since only a variant made names one, so a value read again on its own
//! cycle adds nothing and is taken as the bottom there.
//!
//! Cost: building the summary is O(instructions + edges). A query walks
//! the summary once per value and slot it reaches, O(values + stores) at
//! most.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::cfg::{Aspect, Body, BodyId, Module, Op};
use crate::ir::{PathSeg, RefTarget, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager};
use crate::passes::domain::tags::Tags;
use crate::ty::Mutability;

pub struct TagValues {
    made: FxHashMap<ValueId, Made>,
    stores: FxHashMap<ValueId, Vec<(Vec<PathSeg>, ValueId)>>,
    /// Every slot a `&mut` lends, or that is read or written other than by
    /// name.
    opaque: FxHashSet<ValueId>,
}

/// How a value was made, as far as its variants are concerned.
#[derive(Debug, Clone)]
enum Made {
    Variant { tag: acvus_utils::Astr, payload: Option<ValueId> },
    Parts(Vec<(PathSeg, ValueId)>),
    Joined(Vec<ValueId>),
    /// The part at `path` of the content of `slot`.
    Read { slot: ValueId, path: Vec<PathSeg> },
    /// A reference to the part at `path` of the content of `slot`.
    Lent { slot: ValueId, path: Vec<PathSeg> },
    /// Made by nothing the variants are read off: a parameter, a capture, a
    /// call's result, a read of what another reference names, or any other
    /// operation. It may hold any variant its type has.
    Opaque,
}

impl TagValues {
    /// The variants the part at `path` of `value` may hold; of a reference
    /// to a slot's part, the variants that part may hold. Cost: O(values +
    /// stores).
    pub fn at(&self, value: ValueId, path: &[PathSeg]) -> Tags {
        self.value_at(value, path, &mut FxHashSet::default())
    }

    /// The variants the part at `path` of `slot`'s content may hold. Cost:
    /// O(values + stores).
    pub fn in_slot(&self, slot: ValueId, path: &[PathSeg]) -> Tags {
        self.slot_at(slot, path, &mut FxHashSet::default())
    }

    /// The slot and path a reference `value` lends, where the body lent it
    /// from a slot by name.
    pub fn lent(&self, value: ValueId) -> Option<(ValueId, &[PathSeg])> {
        match self.made.get(&value) {
            Some(Made::Lent { slot, path }) => Some((*slot, path)),
            _ => None,
        }
    }

    fn value_at(&self, value: ValueId, path: &[PathSeg], visiting: &mut FxHashSet<(ValueId, Vec<PathSeg>)>) -> Tags {
        if !visiting.insert((value, path.to_vec())) {
            return Tags::None;
        }
        let made = self.made.get(&value).unwrap_or_else(|| panic!("{value:?} is a value of the body the tags cover"));
        let tags = match made {
            Made::Opaque => Tags::Any,
            Made::Variant { tag, payload } => match (path.split_first(), payload) {
                (None, _) => Tags::Of([*tag].into()),
                (Some((PathSeg::Payload, rest)), Some(payload)) => self.value_at(*payload, rest, visiting),
                // The payload of a variant that has none holds nothing.
                (Some((PathSeg::Payload, _)), None) => Tags::None,
                (Some((seg, _)), _) => panic!("a variant's one part is its payload, not {seg:?}"),
            },
            Made::Parts(parts) => {
                let Some((seg, rest)) = path.split_first() else {
                    panic!("a tuple or an object holds no variant of its own");
                };
                // An object made without a field its grown type holds has no
                // value there (types §1); initck refuses its use whole.
                match parts.iter().find(|(at, _)| at == seg) {
                    Some((_, part)) => self.value_at(*part, rest, visiting),
                    None => Tags::None,
                }
            }
            Made::Joined(incoming) => {
                incoming.iter().fold(Tags::None, |joined, arg| joined.join(&self.value_at(*arg, path, visiting)))
            }
            Made::Read { slot, path: from } => {
                let whole: Vec<PathSeg> = from.iter().chain(path).copied().collect();
                self.slot_at(*slot, &whole, visiting)
            }
            Made::Lent { slot, path: from } => {
                let whole: Vec<PathSeg> = from.iter().chain(path).copied().collect();
                self.slot_at(*slot, &whole, visiting)
            }
        };
        visiting.remove(&(value, path.to_vec()));
        tags
    }

    /// A store at a path the queried one extends writes the part queried;
    /// one deeper than it leaves the variant there as it was.
    fn slot_at(&self, slot: ValueId, path: &[PathSeg], visiting: &mut FxHashSet<(ValueId, Vec<PathSeg>)>) -> Tags {
        if self.opaque.contains(&slot) {
            return Tags::Any;
        }
        let Some(stores) = self.stores.get(&slot) else {
            return Tags::None;
        };
        stores.iter().fold(Tags::None, |joined, (at, value)| match path.strip_prefix(at.as_slice()) {
            Some(rest) => joined.join(&self.value_at(*value, rest, visiting)),
            None => joined,
        })
    }
}

impl Analysis for TagValues {
    type Facts = TagValues;

    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Storage];

    fn compute(_: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> TagValues {
        summary(module.body(body))
    }
}

fn summary(body: &Body) -> TagValues {
    let mut made = FxHashMap::default();
    let mut stores: FxHashMap<ValueId, Vec<(Vec<PathSeg>, ValueId)>> = FxHashMap::default();
    let mut opaque = FxHashSet::default();
    let mut incoming: FxHashMap<ValueId, Vec<ValueId>> = FxHashMap::default();
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            match inst.op() {
                Op::Assign { target: RefTarget::Var(slot), path, value, .. } => {
                    stores.entry(*slot).or_default().push((path.clone(), *value));
                }
                Op::Ref { target: RefTarget::Var(slot), path, mutability } => {
                    if *mutability == Mutability::Mut {
                        opaque.insert(*slot);
                    }
                    if let Some(result) = inst.result() {
                        made.insert(result, Made::Lent { slot: *slot, path: path.clone() });
                    }
                    continue;
                }
                _ => {}
            }
            let Some(result) = inst.result() else {
                continue;
            };
            let of = match inst.op() {
                Op::MakeVariant { tag, payload } => Made::Variant { tag: *tag, payload: *payload },
                Op::MakeTuple { elements } => {
                    Made::Parts(elements.iter().enumerate().map(|(at, element)| (PathSeg::Index(at), *element)).collect())
                }
                Op::MakeObject { fields } => Made::Parts(fields.iter().map(|(name, field)| (PathSeg::Field(*name), *field)).collect()),
                Op::Take { target: RefTarget::Var(slot), path, .. } => Made::Read { slot: *slot, path: path.clone() },
                _ => Made::Opaque,
            };
            made.insert(result, of);
        }
        let terminator = block.terminator();
        for edge in terminator.edges() {
            let params = body.block(edge.to).params();
            for (param, arg) in params.iter().skip(edge.supplied).zip(edge.args) {
                incoming.entry(*param).or_default().push(*arg);
            }
        }
    }
    for (param, args) in incoming {
        made.insert(param, Made::Joined(args));
    }
    // What comes from outside the body, and what a terminator supplies its
    // targets itself, a loop's counter or element, is made by nothing read
    // here.
    let outside = body.params().iter().chain(body.captures()).map(|(_, value)| *value);
    let supplied = body.blocks().flat_map(|(_, block)| block.params().to_vec()).filter(|param| !made.contains_key(param)).collect::<Vec<_>>();
    for value in outside.chain(supplied) {
        made.entry(value).or_insert(Made::Opaque);
    }
    TagValues { made, stores, opaque }
}
