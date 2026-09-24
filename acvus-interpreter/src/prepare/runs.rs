//! Where each aggregate that needs an address lives (RFC-0050 rules 2, 3, 8, 9).
//!
//! This runs once, after the scalar colouring and before the first operation is
//! emitted. The two allocators do not call each other, and that is a decision
//! rather than an omission: three earlier attempts coloured runs inside
//! `assign_slots` and interleaved their emission through `scratch_used`, and the
//! coupling — not slot scarcity — is what each of them foundered on. A body uses
//! far fewer than the registers it may colour, so a run needs nothing of the
//! scalar allocator but the number it stops at.
//!
//! Decision not to ask `analysis::escape` here. That predicate answers whether
//! a value outlives the instruction that hands it on, and rule 3's joins are
//! exactly the instructions it answers yes for: an `Assign` of a block
//! parameter into the storage a web shares with it puts that parameter in the
//! escaped set, so asking it of a web heaps every web this pass exists to
//! place. `Sites` answers the question this pass has to ask instead: is every
//! mention of a member one the emitter has a register form for.

use acvus_mir::analysis::inst_info;
use acvus_mir::ir::{InstKind, Label, MirBody, PathSeg, RefTarget, ValueId};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner, LocalIdOps};
use rustc_hash::{FxHashMap, FxHashSet};

use super::{LiveRange, owns_large};
use crate::code::Slot;
use crate::regs::{MAX_FRAME_SLOTS, MAX_RUN_SLOTS};

/// One register of an aggregate's flat layout (RFC-0050 rule 8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Word {
    /// The field of the run's own type that begins here. A discriminant, a
    /// variant payload and every register inside a nested field begin no field
    /// of that type: `&obj.f` reaches one level, and a deeper path is a
    /// projection of its own.
    pub field: Option<Astr>,
    pub large: bool,
}

/// An aggregate's registers, in the order rule 8 fixes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Layout {
    words: Box<[Word]>,
    tags: Tags,
}

/// The variants a run's type names, which is what decides whether a web's
/// widest layout is a home for a narrower member's tags (`Layout::subsumes`).
///
/// The tag word itself is not numbered here. A run and a heap variant spell one
/// tag the same way — `value::Value::tag` writes both — because a construction
/// holds only the one variant's own type: measured over
/// `acvus-interpreter-test`, every heap `MakeVariant` destination names exactly
/// the variant it wrote and every `Switch` scrutinee names the union, so a
/// position in this list would be a different number at the writer and at the
/// reader.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Tags {
    names: Box<[Astr]>,
}

/// One variant of one type. There is deliberately no way to a tag word other
/// than `Tags::member`: the question "does this type name this variant" is the
/// checker's, already answered, and a `u64` handed around downstream would
/// invite each reader to ask it again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Member {
    name: Astr,
}

impl Member {
    /// The register at offset zero of a run holding this variant, which is the
    /// register `value::Value::variant` writes into a heap variant's header.
    pub fn register(self) -> crate::value::Value {
        crate::value::Value::tag(self.name)
    }

    pub fn word(self) -> u64 {
        self.register().bits()
    }
}

impl Tags {
    fn of<'v, I>(variants: I, interner: &Interner) -> Tags
    where
        I: Iterator<Item = &'v Astr>,
    {
        let mut names: Vec<Astr> = variants.copied().collect();
        names.sort_by(|a, b| interner.resolve(*a).cmp(interner.resolve(*b)));
        Tags {
            names: names.into_boxed_slice(),
        }
    }

    pub fn holds(&self, name: Astr) -> bool {
        self.names.contains(&name)
    }

    pub fn member(&self, name: Astr) -> Option<Member> {
        self.holds(name).then_some(Member { name })
    }

    pub fn names(&self) -> &[Astr] {
        &self.names
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Whether the type being laid is the run's own or one of its fields'. Only the
/// run's own fields are named in the layout.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Outer,
    Inner,
}

impl Layout {
    /// `None` where rule 8 fixes no order for `ty`.
    pub fn of(ty: &Ty, interner: &Interner) -> Option<Layout> {
        let mut words = Vec::new();
        lay(ty, interner, Level::Outer, &mut words)?;
        let tags = match ty {
            Ty::Enum { variants, .. } => Tags::of(variants.keys(), interner),
            Ty::Result(..) => Tags::of(
                [interner.intern("Ok"), interner.intern("Err")].iter(),
                interner,
            ),
            _ => Tags::default(),
        };
        Some(Layout {
            words: words.into_boxed_slice(),
            tags,
        })
    }

    pub fn tags(&self) -> &Tags {
        &self.tags
    }

    /// # Panics
    /// The layout is a structural object's, which has no payload.
    pub fn payload(&self) -> u16 {
        assert!(!self.tags.is_empty(), "only an enum's run has a payload");
        1
    }

    /// Whether one construction can write this whole layout. Every value a
    /// `MakeObject` or a `MakeVariant` names is one register, so a layout with
    /// a register no such value reaches is one this pass cannot fill.
    /// The fields of the run's own type, with the register each begins.
    pub fn fields(&self) -> impl Iterator<Item = (u16, Astr)> + '_ {
        self.words.iter().enumerate().filter_map(|(at, word)| {
            word.field
                .map(|name| (u16::try_from(at).expect("a layout is bounded"), name))
        })
    }

    /// Whether this layout is a home for a value of `narrow`'s type too: the
    /// members of one web carry the union type's variants and fields in
    /// different widths, because a `MakeVariant` result is typed by the one
    /// variant it wrote and the join it reaches carries the union (RFC-0041).
    pub fn subsumes(&self, narrow: &Layout) -> bool {
        if narrow.words.len() > self.words.len() {
            return false;
        }
        let tags = narrow
            .tags
            .names()
            .iter()
            .all(|name| self.tags.holds(*name));
        let fields = narrow.fields().all(|(_, name)| self.field(name).is_some());
        tags && fields && narrow.tags.is_empty() == self.tags.is_empty()
    }

    pub fn lowerable(&self) -> bool {
        match self.tags.is_empty() {
            true => self.words.iter().all(|word| word.field.is_some()),
            false => self.words.len() == 2,
        }
    }

    pub fn len(&self) -> u16 {
        u16::try_from(self.words.len()).expect("a layout is bounded by the registers a run holds")
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    #[cfg(any(test, feature = "tooling"))]
    pub fn large(&self, at: u16) -> bool {
        self.words[usize::from(at)].large
    }

    /// The register `name`'s field begins at, for a projection into the middle
    /// of a run.
    pub fn field(&self, name: Astr) -> Option<u16> {
        self.words
            .iter()
            .position(|word| word.field == Some(name))
            .map(|at| u16::try_from(at).expect("a layout is bounded by a run's registers"))
    }

    /// The registers whose `Large` the frame's sweep releases.
    pub fn releases(&self) -> impl Iterator<Item = u16> + '_ {
        self.words
            .iter()
            .enumerate()
            .filter(|(_, word)| word.large)
            .map(|(at, _)| u16::try_from(at).expect("a layout is bounded by a run's registers"))
    }
}

fn lay(ty: &Ty, interner: &Interner, level: Level, out: &mut Vec<Word>) -> Option<()> {
    match ty {
        Ty::Object(obj) => {
            for (name, field) in crate::layout::sorted_fields(interner, obj) {
                let at = out.len();
                lay(field, interner, Level::Inner, out)?;
                if let Level::Outer = level {
                    out[at].field = Some(*name);
                }
            }
        }
        Ty::Tuple(items) => {
            for item in items {
                lay(item, interner, Level::Inner, out)?;
            }
        }
        Ty::Enum { variants, .. } => {
            out.push(Word {
                field: None,
                large: false,
            });
            widest(
                variants.values().filter_map(|v| v.as_deref()),
                interner,
                out,
            )?;
        }
        Ty::Result(ok, err) => {
            out.push(Word {
                field: None,
                large: false,
            });
            widest([ok.as_ref(), err.as_ref()].into_iter(), interner, out)?;
        }
        // Rule 9 and RFC-0039: an option is its payload, at the payload's width,
        // and the run's first register is `Kind::None` for `None`. That is why
        // the payload's own first register is the option's, and why no register
        // of the layout is spent on a discriminant.
        Ty::Option(inner) => lay(inner, interner, level, out)?,
        Ty::Slice(_) | Ty::Str => return None,
        _ => out.push(Word {
            field: None,
            large: owns_large(ty),
        }),
    }
    Some(())
}

/// The variants laid over one another: the widest payload's registers, each
/// owning a `Large` where any variant's does.
///
/// RFC-0052 rule 5 is what makes one register enough for payloads that disagree
/// in class — the register is whole-typed and `Release` decides by kind, so there
/// is no conditional drop for this to encode.
fn widest<'t, I>(payloads: I, interner: &Interner, out: &mut Vec<Word>) -> Option<()>
where
    I: Iterator<Item = &'t Ty>,
{
    let mut over: Vec<Word> = Vec::new();
    for payload in payloads {
        let mut words = Vec::new();
        lay(payload, interner, Level::Inner, &mut words)?;
        for (at, word) in words.into_iter().enumerate() {
            match over.get_mut(at) {
                Some(held) => held.large |= word.large,
                None => over.push(word),
            }
        }
    }
    out.extend(over);
    Some(())
}

#[derive(Clone, Debug)]
pub struct Run {
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    pub var: ValueId,
    pub base: Slot,
    pub layout: Layout,
    pub members: Vec<ValueId>,
    pub projections: Vec<ValueId>,
}

/// Where every addressed aggregate of one body lives.
#[derive(Clone, Debug, Default)]
pub struct RunPlan {
    pub runs: Vec<Run>,
    /// The registers all of the runs together take above the scalar ones.
    pub total: u16,
    /// The aggregates that did not fit, realized on the heap instead (rule 4).
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    pub heaped: Vec<ValueId>,
}

impl RunPlan {
    pub fn of(&self, var: ValueId) -> Option<&Run> {
        self.runs.iter().find(|run| run.members.contains(&var))
    }

    pub fn projected(&self, var: ValueId) -> Option<&Run> {
        self.runs.iter().find(|run| run.projections.contains(&var))
    }

    pub fn registers(&self) -> impl Iterator<Item = Slot> + '_ {
        self.runs
            .iter()
            .flat_map(|run| (0..run.layout.len()).map(|at| run.base + at))
    }
}

struct Candidate {
    var: ValueId,
    layout: Layout,
    range: LiveRange,
    depth: u32,
    members: Vec<ValueId>,
    projections: Vec<ValueId>,
}

/// The registers one placed run holds over its live range.
struct Held {
    range: LiveRange,
    at: u16,
    len: u16,
}

/// The aggregates of `body` that need an address, placed in the registers above
/// `base`.
///
/// `ranges` is indexed by `ValueId::to_raw`, which is what the scalar colouring
/// indexes its own tables by; the two agree because this is handed the table
/// that colouring built.
pub(crate) fn plan(
    body: &MirBody,
    labels: &FxHashMap<Label, u32>,
    base: Slot,
    ranges: &[Option<LiveRange>],
    interner: &Interner,
    written_by_a_call: &FxHashSet<ValueId>,
) -> RunPlan {
    let mut candidates = candidates(body, labels, ranges, interner, written_by_a_call);
    let mut heaped = Vec::new();
    loop {
        if let Some(runs) = place(&candidates, base) {
            let total = runs
                .iter()
                .map(|run| run.base + run.layout.len() - base)
                .max()
                .unwrap_or(0);
            return RunPlan {
                runs,
                total,
                heaped,
            };
        }
        heaped.extend(candidates.remove(spill_first(&candidates)).members);
    }
}

/// Which candidate gives its run up to the heap. What a loop touches stays in
/// the frame, so the shallowest depth goes first; at one depth the longest live
/// range goes, because it holds registers for the longest.
fn spill_first(candidates: &[Candidate]) -> usize {
    candidates
        .iter()
        .enumerate()
        .min_by_key(|(at, c)| {
            (
                c.depth,
                std::cmp::Reverse(c.range.hi - c.range.lo),
                c.var.to_raw(),
                *at,
            )
        })
        .map(|(at, _)| at)
        .expect("an empty candidate set takes no registers, so it always places")
}

/// Two values rule 3 gives one home. Union is symmetric, so neither side is
/// the target of the other.
#[derive(Clone, Copy)]
struct Join {
    one: ValueId,
    other: ValueId,
}

struct Edges<'a> {
    body: &'a MirBody,
    labels: &'a FxHashMap<Label, u32>,
}

impl Edges<'_> {
    fn params(&self, label: &Label) -> &[ValueId] {
        let at = *self
            .labels
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {label:?}")) as usize;
        let InstKind::BlockLabel { params, .. } = &self.body.insts[at].kind else {
            panic!("a jump names {label:?}, whose instruction is not a block label")
        };
        params
    }

    fn carried(&self, params: &[ValueId], args: &[ValueId], out: &mut Vec<Join>) {
        out.extend(params.iter().zip(args).map(|(param, arg)| Join {
            one: *param,
            other: *arg,
        }));
    }

    fn edge(&self, label: &Label, args: &[ValueId], out: &mut Vec<Join>) {
        self.carried(self.params(label), args, out);
    }

    fn joins(&self) -> Vec<Join> {
        let mut out = Vec::new();
        for inst in &self.body.insts {
            match &inst.kind {
                InstKind::Assign {
                    target,
                    path,
                    value,
                    ..
                } if path.is_empty() => {
                    if let Some(storage) = inst_info::storage(target) {
                        out.push(Join {
                            one: storage,
                            other: *value,
                        });
                    }
                }
                InstKind::Jump { label, args } => self.edge(label, args, &mut out),
                InstKind::JumpIf {
                    then_label,
                    then_args,
                    else_label,
                    else_args,
                    ..
                } => {
                    self.edge(then_label, then_args, &mut out);
                    self.edge(else_label, else_args, &mut out);
                }
                InstKind::Switch { arms, default, .. } => {
                    for (_, label, args) in arms {
                        self.edge(label, args, &mut out);
                    }
                    if let Some((label, args)) = default {
                        self.edge(label, args, &mut out);
                    }
                }
                kind @ (InstKind::For { .. } | InstKind::ForParts { .. }) => {
                    let acvus_mir::ir::Traversal {
                        source,
                        body,
                        body_args,
                        exit,
                        exit_trip,
                        exit_args,
                    } = acvus_mir::ir::traversal(kind).expect("a `For` or a `ForParts`");
                    let leaving = exit_trip.carried_params(self.params(&exit));
                    self.carried(leaving, exit_args, &mut out);
                    let carried = source.carried_params(self.params(&body));
                    self.carried(carried, &body_args, &mut out);
                }
                _ => {}
            }
        }
        out
    }
}

/// Where a web member is mentioned, and whether this pass lowers that mention.
///
/// The default arm refuses, so an instruction kind added to the IR takes its
/// web to the heap rather than reaching a lowering that was never written for
/// it. That direction is the whole of the soundness argument here: a member
/// whose mention is refused has no run, and a member with a run is mentioned
/// only where the emitter has an arm.
struct Sites<'a> {
    class_of: &'a [usize],
    web: &'a FxHashSet<usize>,
    projected: &'a FxHashMap<ValueId, usize>,
    written_by_a_call: &'a FxHashSet<ValueId>,
    refused: FxHashSet<usize>,
    read: FxHashSet<usize>,
}

impl Sites<'_> {
    fn web_of(&self, value: ValueId) -> Option<usize> {
        let root = self.class_of[value.to_raw()];
        self.web.contains(&root).then_some(root)
    }

    /// A mention this pass has no register form for.
    fn refuse(&mut self, value: ValueId) {
        if let Some(root) = self.web_of(value) {
            self.refused.insert(root);
        }
        if let Some(root) = self.projected.get(&value) {
            self.refused.insert(*root);
        }
    }

    fn refuse_target(&mut self, target: &RefTarget) {
        if let Some(storage) = inst_info::storage(target) {
            self.refuse(storage);
        }
        if let RefTarget::Through(through) = target {
            self.refuse(*through);
        }
    }

    /// The web a tag test or a payload unwrap reads out of the run's own
    /// registers: the one `value` is a member of, or the one it projects onto.
    /// This is `prepare::Prepare::run_tag`'s question, asked of the web before
    /// the run is placed.
    fn reads_a_register(&self, value: ValueId) -> Option<usize> {
        self.web_of(value)
            .or_else(|| self.projected.get(&value).copied())
    }

    /// The web a place reaches: the storage it names, or the one the
    /// reference it walks through projects onto.
    fn reached(&self, target: &RefTarget) -> Option<usize> {
        match target {
            RefTarget::Var(storage) | RefTarget::Param(storage) => self.web_of(*storage),
            RefTarget::Through(through) => self.projected.get(through).copied(),
        }
    }

    fn observe(&mut self, kind: &InstKind) {
        match kind {
            InstKind::MakeObject { dst, fields } => {
                if self.web_of(*dst).is_none() {
                    self.refuse(*dst);
                }
                for (_, value) in fields {
                    self.refuse(*value);
                }
            }
            InstKind::MakeVariant { dst, payload, .. } => {
                if self.web_of(*dst).is_none() {
                    self.refuse(*dst);
                }
                if let Some(value) = payload {
                    self.refuse(*value);
                }
            }
            InstKind::Assign {
                target,
                path,
                value,
                ..
            } => {
                let joined = path.is_empty()
                    && self.reached(target).is_some()
                    && self.reached(target) == self.web_of(*value);
                if !joined {
                    self.refuse_target(target);
                    self.refuse(*value);
                }
            }
            InstKind::Ref { target, path, .. } => {
                if !path.is_empty() || self.reached(target).is_none() {
                    self.refuse_target(target);
                }
            }
            InstKind::Take {
                dst, target, path, ..
            } => {
                let read = matches!(path.as_slice(), [PathSeg::Field(_) | PathSeg::Payload])
                    && self.reached(target).is_some();
                if !read {
                    self.refuse_target(target);
                    self.refuse(*dst);
                }
            }
            InstKind::Drop { .. }
            | InstKind::Switch { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::BlockLabel { .. } => {}
            InstKind::TestVariant { dst, src, .. } | InstKind::UnwrapVariant { dst, src } => {
                match self.reads_a_register(*src) {
                    Some(root) => {
                        self.read.insert(root);
                    }
                    None => self.refuse(*src),
                }
                self.refuse(*dst);
            }
            InstKind::FunctionCall { dst, .. } if self.written_by_a_call.contains(dst) => {
                for value in inst_info::uses(kind) {
                    self.refuse(value);
                }
            }
            other => {
                for value in inst_info::uses(other) {
                    self.refuse(value);
                }
                for def in inst_info::defs(other) {
                    self.refuse(def);
                }
            }
        }
    }
}

fn candidates(
    body: &MirBody,
    labels: &FxHashMap<Label, u32>,
    ranges: &[Option<LiveRange>],
    interner: &Interner,
    written_by_a_call: &FxHashSet<ValueId>,
) -> Vec<Candidate> {
    let depths = loop_depths(body, labels);
    let values = body.val_factory.len();

    let entry: FxHashSet<ValueId> = body
        .params
        .iter()
        .chain(&body.captures)
        .map(|(_, id)| *id)
        .chain(body.order_param)
        .collect();

    let mut classes = super::Classes::new(values);
    for Join { one, other } in (Edges { body, labels }).joins() {
        let (one, other) = (classes.find(one.to_raw()), classes.find(other.to_raw()));
        if one != other {
            classes.unite(one, other);
        }
    }
    let class_of: Vec<usize> = (0..values).map(|value| classes.find(value)).collect();

    let mut members: FxHashMap<usize, Vec<ValueId>> = FxHashMap::default();
    for value in 0..values {
        if ranges[value].is_none() {
            continue;
        }
        let id = ValueId::from_raw(value);
        let ty = body
            .val_types
            .get(&id)
            .unwrap_or_else(|| panic!("no type for value {id:?}"));
        if !laid_whole(ty) {
            continue;
        }
        if entry.contains(&id) {
            continue;
        }
        members.entry(class_of[value]).or_default().push(id);
    }

    let mut layouts: FxHashMap<usize, Layout> = FxHashMap::default();
    members.retain(|root, held| {
        let laid: Option<Vec<Layout>> = held
            .iter()
            .map(|id| Layout::of(&body.val_types[id], interner))
            .collect();
        let Some(laid) = laid else { return false };
        let widest = laid
            .iter()
            .max_by_key(|layout| (layout.len(), layout.tags.len()))
            .expect("a web is the members that made it");
        if widest.is_empty()
            || !widest.lowerable()
            || laid.iter().any(|held| !widest.subsumes(held))
        {
            return false;
        }
        layouts.insert(*root, widest.clone());
        true
    });

    let web: FxHashSet<usize> = members.keys().copied().collect();
    let mut projected: FxHashMap<ValueId, usize> = FxHashMap::default();
    for inst in &body.insts {
        if let InstKind::Ref {
            dst, target, path, ..
        } = &inst.kind
            && path.is_empty()
            && let Some(storage) = inst_info::storage(target)
            && let Some(root) = web.get(&class_of[storage.to_raw()])
        {
            projected.insert(*dst, *root);
        }
    }

    let mut sites = Sites {
        class_of: &class_of,
        web: &web,
        projected: &projected,
        written_by_a_call: written_by_a_call,
        refused: FxHashSet::default(),
        read: projected.values().copied().collect(),
    };
    for inst in &body.insts {
        sites.observe(&inst.kind);
    }
    for id in &entry {
        sites.refuse(*id);
    }
    // A `For` defines the parameters it supplies itself — the element and the
    // counter, and the trip count on an exit edge that defines one — as the
    // machine writes them, which no instruction's `defs` names; an element
    // is written as a heap value, so it has no run.
    let edges = Edges { body, labels };
    for inst in &body.insts {
        if let Some(acvus_mir::ir::Traversal {
            source,
            body,
            exit,
            exit_trip,
            ..
        }) = acvus_mir::ir::traversal(&inst.kind)
        {
            let params = edges.params(&body);
            for supplied in &params[..params.len() - source.carried_params(params).len()] {
                sites.refuse(*supplied);
            }
            if let Some(trip) = exit_trip.trip_param(edges.params(&exit)) {
                sites.refuse(trip);
            }
        }
    }
    let (refused, read) = (sites.refused, sites.read);

    let mut out: Vec<Candidate> = Vec::new();
    for (root, mut held) in members {
        if refused.contains(&root) {
            continue;
        }
        let projections: Vec<ValueId> = projected
            .iter()
            .filter(|(_, at)| **at == root)
            .map(|(dst, _)| *dst)
            .collect();
        let lands_from_a_call = held.iter().any(|id| written_by_a_call.contains(id));
        if !read.contains(&root) && !lands_from_a_call {
            continue;
        }
        held.sort_by_key(|id| id.to_raw());
        let range = held
            .iter()
            .chain(&projections)
            .map(|id| {
                ranges[id.to_raw()]
                    .unwrap_or_else(|| panic!("value {id:?} is in a web and has no live range"))
            })
            .reduce(|held, one| held.joined(one))
            .expect("a web is the members that made it");
        out.push(Candidate {
            var: held[0],
            layout: layouts.remove(&root).expect("every web was laid"),
            range,
            depth: depths[range.lo],
            members: held,
            projections,
        });
    }
    out.sort_by_key(|candidate| candidate.var.to_raw());
    out
}

/// Whether rule 8 lays this type as a run of its own. A nested aggregate is
/// laid inline inside one, and a `Tuple` and an `Option` are laid as fields;
/// an enum, a `Result` and a structural object are a run's own type, because
/// those three have the construction and the dispatch this pass lowers.
fn laid_whole(ty: &Ty) -> bool {
    matches!(ty, Ty::Enum { .. } | Ty::Result(..) | Ty::Object(_))
}

/// RFC-0050 rule 2's placement order: deepest loop first, then live-range start.
/// `place` gives the lowest free registers to whichever candidate it reaches
/// first, so this order is what decides who gets them.
fn placement_order(candidates: &[Candidate]) -> Vec<&Candidate> {
    let mut order: Vec<&Candidate> = candidates.iter().collect();
    order.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.depth),
            c.range.lo,
            c.range.hi,
            c.var.to_raw(),
        )
    });
    order
}

/// A linear scan in `placement_order`. A run takes the lowest registers free
/// over its own range, so two runs whose ranges do not overlap share registers
/// and a run never straddles a live one.
fn place(candidates: &[Candidate], base: Slot) -> Option<Vec<Run>> {
    let mut held: Vec<Held> = Vec::new();
    let mut runs = Vec::with_capacity(candidates.len());
    for candidate in placement_order(candidates) {
        let len = candidate.layout.len();
        if len > MAX_RUN_SLOTS {
            return None;
        }
        let free = |at: u16| {
            held.iter().all(|h| {
                !h.range.overlaps(candidate.range) || at + len <= h.at || h.at + h.len <= at
            })
        };
        let last = MAX_RUN_SLOTS - len;
        let mut at = 0u16;
        while !free(at) {
            if at == last {
                return None;
            }
            at += 1;
        }
        if usize::from(base) + usize::from(at) + usize::from(len) > usize::from(MAX_FRAME_SLOTS) {
            return None;
        }
        held.push(Held {
            range: candidate.range,
            at,
            len,
        });
        runs.push(Run {
            var: candidate.var,
            base: base + at,
            layout: candidate.layout.clone(),
            members: candidate.members.clone(),
            projections: candidate.projections.clone(),
        });
    }
    Some(runs)
}

/// How many back edges enclose each instruction. A jump to a label that lies
/// behind it closes a loop over everything between the two.
fn loop_depths(body: &MirBody, labels: &FxHashMap<Label, u32>) -> Vec<u32> {
    let mut depth = vec![0u32; body.insts.len()];
    for (at, inst) in body.insts.iter().enumerate() {
        let targets: Vec<Label> = match &inst.kind {
            InstKind::Jump { label, .. } => vec![*label],
            InstKind::JumpIf {
                then_label,
                else_label,
                ..
            } => vec![*then_label, *else_label],
            InstKind::Switch { arms, default, .. } => arms
                .iter()
                .map(|(_, label, _)| *label)
                .chain(default.iter().map(|(label, _)| *label))
                .collect(),
            kind @ (InstKind::For { .. } | InstKind::ForParts { .. }) => {
                let traversal = acvus_mir::ir::traversal(kind).expect("a `For` or a `ForParts`");
                vec![traversal.body, traversal.exit]
            }
            InstKind::Diamond {
                then_label,
                else_label,
                join,
                ..
            } => vec![*then_label, *else_label, *join],
            _ => Vec::new(),
        };
        for label in targets {
            let head = *labels
                .get(&label)
                .unwrap_or_else(|| panic!("unknown label {label:?}"))
                as usize;
            if head <= at {
                for entry in &mut depth[head..=at] {
                    *entry += 1;
                }
            }
        }
    }
    depth
}

#[cfg(test)]
mod tests {
    use acvus_mir::ty::ObjectTy;
    use acvus_utils::{Interner, QualifiedRef};

    use crate::regs::MAX_SCALAR_SLOTS;

    use super::*;

    fn object(i: &Interner, fields: &[(&str, Ty)]) -> Ty {
        Ty::Object(ObjectTy::written(
            fields
                .iter()
                .map(|(name, ty)| (i.intern(name), ty.clone()))
                .collect(),
        ))
    }

    fn enum_of(i: &Interner, variants: &[(&str, Option<Ty>)]) -> Ty {
        Ty::Enum {
            name: QualifiedRef::root(i.intern("E")),
            variants: variants
                .iter()
                .map(|(name, ty)| (i.intern(name), ty.clone().map(Box::new)))
                .collect(),
            home: acvus_mir::ty::Home::NONE,
        }
    }

    fn layout(ty: &Ty, i: &Interner) -> Layout {
        Layout::of(ty, i).expect("rule 8 fixes an order for this type")
    }

    /// Rule 8's one field order, at the only contract that can hold both
    /// artifacts: a field set interned in the reverse of its string order, laid
    /// by this pass and ordered by `layout::encode`'s own comparison, comes out
    /// the same both ways. Interning order is what the two would disagree on,
    /// so the shuffle is what makes the test discriminate.
    #[test]
    fn a_run_and_the_canonical_encoding_order_a_shuffled_field_set_alike() {
        let i = Interner::new();
        let names = ["zeta", "alpha", "mu", "beta"];
        let ty = object(
            &i,
            &names
                .iter()
                .map(|name| (*name, Ty::I64))
                .collect::<Vec<(&str, Ty)>>(),
        );
        let Ty::Object(obj) = &ty else {
            panic!("an object type")
        };
        let laid = layout(&ty, &i);
        let encoded: Vec<&str> = crate::layout::sorted_fields(&i, obj)
            .iter()
            .map(|(name, _)| i.resolve(**name))
            .collect();
        assert_eq!(
            encoded,
            vec!["alpha", "beta", "mu", "zeta"],
            "the canonical encoding is in string order"
        );
        for (at, name) in encoded.iter().enumerate() {
            assert_eq!(
                laid.field(i.intern(name)),
                Some(u16::try_from(at).expect("four fields")),
                "field {name} of the run"
            );
        }
    }

    #[test]
    fn a_structural_object_is_its_fields_in_string_order() {
        let i = Interner::new();
        let ty = object(&i, &[("b", Ty::I64), ("a", Ty::String)]);
        let laid = layout(&ty, &i);
        assert_eq!(laid.len(), 2);
        let (a, b) = (i.intern("a"), i.intern("b"));
        assert_eq!(laid.field(a), Some(0), "`a` sorts before `b` as a string");
        assert_eq!(laid.field(b), Some(1));
        assert!(laid.large(0), "a String field owns a Large");
        assert!(!laid.large(1));
        assert_eq!(laid.releases().collect::<Vec<u16>>(), vec![0]);
    }

    /// Rule 8 as corrected gives every object type one order — its field
    /// names sorted as strings — and a struct's declaration order is not in
    /// `ObjectTy` to contradict it. So a declared struct lays like a written
    /// one, which is what lets an extern's `-> S` result take a run.
    #[test]
    fn a_declared_struct_lays_in_string_order_like_any_other_object() {
        let i = Interner::new();
        let ty = Ty::Object(ObjectTy::declared(
            QualifiedRef::root(i.intern("Point")),
            [(i.intern("y"), Ty::I64), (i.intern("x"), Ty::String)]
                .into_iter()
                .collect(),
        ));
        let laid = layout(&ty, &i);
        assert_eq!(laid.len(), 2);
        assert_eq!(laid.field(i.intern("x")), Some(0));
        assert_eq!(laid.field(i.intern("y")), Some(1));
        assert!(laid.large(0), "a String field owns a Large");
        assert!(laid.lowerable(), "one register per field takes a run");
    }

    #[test]
    fn a_nested_aggregate_is_inline_at_its_offset() {
        let i = Interner::new();
        let inner = object(&i, &[("p", Ty::String), ("q", Ty::I64)]);
        let ty = object(&i, &[("a", inner), ("z", Ty::Bool)]);
        let laid = layout(&ty, &i);
        assert_eq!(laid.len(), 3, "two inner registers and one outer");
        assert_eq!(laid.field(i.intern("a")), Some(0));
        assert_eq!(laid.field(i.intern("z")), Some(2));
        assert_eq!(
            laid.field(i.intern("p")),
            None,
            "an inner field names no field of the outer type"
        );
        assert_eq!(laid.releases().collect::<Vec<u16>>(), vec![0]);
    }

    #[test]
    fn an_enum_is_a_tag_and_its_widest_payload() {
        let i = Interner::new();
        let pair = Ty::Tuple(vec![Ty::I64, Ty::I64]);
        let ty = enum_of(
            &i,
            &[("A", Some(Ty::String)), ("B", Some(pair)), ("C", None)],
        );
        let laid = layout(&ty, &i);
        assert_eq!(laid.len(), 3, "a tag and the two of the widest payload");
        assert!(!laid.large(0), "a tag owns no Large");
        assert!(
            laid.large(1),
            "the first payload register is whole-typed over the variants that disagree"
        );
        assert_eq!(laid.releases().collect::<Vec<u16>>(), vec![1]);
    }

    #[test]
    fn an_option_is_its_payload_flat() {
        let i = Interner::new();
        let inner = object(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
        let laid = layout(&Ty::Option(Box::new(inner)), &i);
        assert_eq!(laid.len(), 2, "no register is spent on a discriminant");
        assert_eq!(laid.field(i.intern("a")), Some(0));
    }

    #[test]
    fn a_slice_has_no_layout() {
        let i = Interner::new();
        assert_eq!(Layout::of(&Ty::Slice(Box::new(Ty::I64)), &i), None);
    }

    fn candidate(var: usize, lo: usize, hi: usize, len: u16, depth: u32) -> Candidate {
        let i = Interner::new();
        let ty = Ty::Tuple(vec![Ty::I64; usize::from(len)]);
        let id = ValueId::from_raw(var);
        Candidate {
            var: id,
            layout: Layout::of(&ty, &i).expect("a tuple of words has a layout"),
            range: LiveRange { lo, hi },
            depth,
            members: vec![id],
            projections: Vec::new(),
        }
    }

    #[test]
    fn a_run_begins_where_the_scalars_end() {
        let placed = place(&[candidate(0, 0, 5, 2, 0)], 9).expect("one run fits");
        assert_eq!(placed[0].base, 9);
        assert_eq!(placed[0].layout.len(), 2);
    }

    #[test]
    fn two_runs_whose_ranges_are_apart_share_registers() {
        let placed =
            place(&[candidate(0, 0, 3, 2, 0), candidate(1, 4, 9, 2, 0)], 9).expect("two runs fit");
        assert_eq!(placed[0].base, placed[1].base);
    }

    #[test]
    fn two_runs_whose_ranges_overlap_do_not() {
        let placed =
            place(&[candidate(0, 0, 9, 2, 0), candidate(1, 4, 9, 3, 0)], 9).expect("two runs fit");
        assert_eq!(placed[0].base, 9);
        assert_eq!(placed[1].base, 11);
    }

    /// The bound is the frame's, not the region's alone: the whole region fits
    /// above a low base and does not above a high one, because a body's scalars
    /// and its runs share the 320 registers one frame holds.
    #[test]
    fn a_run_places_by_what_the_frame_holds_above_its_base() {
        let widest = || candidate(0, 0, 9, MAX_RUN_SLOTS, 0);
        assert_eq!(
            place(&[widest()], 9).expect("the region fits above nine scalars")[0].base,
            9
        );
        assert!(
            place(&[widest()], MAX_SCALAR_SLOTS + 1).is_none(),
            "the region does not fit above a full scalar frame"
        );
    }

    fn base_of(runs: &[Run], var: usize) -> Slot {
        runs.iter()
            .find(|run| run.var == ValueId::from_raw(var))
            .expect("every candidate that placed has a run")
            .base
    }

    /// Rule 2's order, at the placement's contract: the outer scope's aggregate
    /// is live first and would take the low registers in live-range order, and
    /// the inner loop's takes them instead.
    #[test]
    fn an_inner_loop_s_run_is_placed_before_the_outer_scope_s() {
        let outer = candidate(0, 0, 9, 2, 0);
        let inner = candidate(1, 2, 6, 2, 1);
        let placed = place(&[outer, inner], 9).expect("two runs fit");
        assert_eq!(base_of(&placed, 1), 9, "the inner loop's run");
        assert_eq!(base_of(&placed, 0), 11, "the outer scope's run");
    }

    /// The runs one loop touches are placed first and reuse registers among
    /// themselves, so two inner runs whose ranges are apart share the lowest
    /// registers and the outer scope's run sits above both.
    #[test]
    fn two_inner_runs_apart_in_range_share_registers_ahead_of_the_outer() {
        let outer = candidate(0, 0, 9, 2, 0);
        let first = candidate(1, 2, 4, 2, 1);
        let second = candidate(2, 5, 7, 2, 1);
        let placed = place(&[outer, first, second], 9).expect("three runs fit");
        assert_eq!(base_of(&placed, 1), 9);
        assert_eq!(
            base_of(&placed, 2),
            9,
            "apart in range, so the same registers"
        );
        assert_eq!(base_of(&placed, 0), 11);
    }

    #[test]
    fn the_shallowest_loop_gives_its_run_up_first() {
        let held = [candidate(0, 0, 9, 2, 1), candidate(1, 0, 9, 2, 0)];
        assert_eq!(held[spill_first(&held)].depth, 0);
    }

    #[test]
    fn at_one_depth_the_longest_live_range_gives_its_run_up_first() {
        let held = [candidate(0, 0, 2, 2, 1), candidate(1, 0, 9, 2, 1)];
        assert_eq!(held[spill_first(&held)].var, ValueId::from_raw(1));
    }

    #[test]
    fn a_member_exists_for_a_held_name_and_words_it_as_the_names_own_bits() {
        let i = Interner::new();
        let ty = enum_of(&i, &[("Zed", Some(Ty::I64)), ("Alpha", None)]);
        let laid = layout(&ty, &i);

        for name in ["Zed", "Alpha"] {
            let tag = i.intern(name);
            let member = laid
                .tags()
                .member(tag)
                .expect("the type names this variant");
            assert_eq!(member.word(), crate::value::Value::tag(tag).bits());
        }
        assert!(laid.tags().member(i.intern("Omega")).is_none());
    }

    /// RFC-0050 rule 8 says the run and the heap realization are one layout, and
    /// this is the contract both reach it through: a run of `E::B(i64)` and a
    /// heap `Value::variant` of the same tag are compared register for
    /// register.
    ///
    /// The two writers are `prepare::lay_variant`, which writes
    /// `Member::register` at the run's base, and `value::Value::variant`, which
    /// writes `Value::tag` at the header's first register. Give them different
    /// numberings and only this assertion fails.
    #[test]
    fn a_heap_variant_and_a_run_of_one_enum_are_the_same_words() {
        let i = Interner::new();
        let ty = enum_of(
            &i,
            &[
                ("Zed", Some(Ty::I64)),
                ("Alpha", Some(Ty::I64)),
                ("Mu", Some(Ty::I64)),
            ],
        );
        let laid = layout(&ty, &i);
        assert_eq!(
            usize::from(laid.len()),
            acvus_extern::Variant::<()>::WIDTH,
            "a run of an enum is the registers a heap variant holds"
        );
        assert_eq!(
            usize::from(laid.payload()),
            1,
            "the payload follows the tag in both"
        );

        for name in ["Zed", "Alpha", "Mu"] {
            let tag = i.intern(name);
            let run = laid
                .tags()
                .member(tag)
                .expect("the type names this variant")
                .register();
            let heap = crate::value::Value::variant(tag, None);
            // SAFETY: `Value::variant` erased a variant.
            let held = unsafe { heap.as_variant() };
            assert_eq!(run.kind(), held.tag().kind(), "the tag register's kind");
            assert_eq!(run.bits(), held.tag().bits(), "the tag register's word");
            assert_eq!(
                held.payload().kind(),
                crate::value::Kind::Undef,
                "a tag that carries nothing leaves rule 8's Undef"
            );
        }
    }

    /// The sibling of the test above for the one variant type whose tags the
    /// language names rather than a declaration. `Ok` and `Err` reach the tag
    /// register through `Member::register` and `Value::tag` exactly as a
    /// declared variant's name does, so a `Result` and an enum are one layout.
    #[test]
    fn a_heap_result_and_a_run_of_one_result_are_the_same_words() {
        let i = Interner::new();
        let ty = Ty::Result(Box::new(Ty::I64), Box::new(Ty::String));
        let laid = layout(&ty, &i);
        assert_eq!(
            usize::from(laid.len()),
            acvus_extern::Variant::<()>::WIDTH,
            "a run of a Result is the registers a heap variant holds"
        );
        assert_eq!(usize::from(laid.payload()), 1);

        for name in ["Ok", "Err"] {
            let tag = i.intern(name);
            let run = laid
                .tags()
                .member(tag)
                .expect("a Result names this side")
                .register();
            // SAFETY: an integer word owns nothing.
            let payload = unsafe { acvus_extern::Owned::from_value(acvus_extern::Holding::new(), crate::value::Value::int(7)) };
            let heap = crate::value::Value::variant(tag, Some(payload));
            // SAFETY: `Value::variant` erased a variant.
            let held = unsafe { heap.as_variant() };
            assert_eq!(run.kind(), held.tag().kind(), "the tag register's kind");
            assert_eq!(run.bits(), held.tag().bits(), "the tag register's word");
            assert_eq!(held.payload().as_int(), 7);
        }
    }

    /// The whole pass at its contract: two aggregates too wide to share a frame
    /// leave one on the heap, and the one that stays is the one in the loop.
    #[test]
    fn an_aggregate_past_the_frame_goes_to_the_heap() {
        let held = vec![
            candidate(0, 0, 9, MAX_RUN_SLOTS - 1, 0),
            candidate(1, 0, 9, MAX_RUN_SLOTS - 1, 1),
        ];
        let mut candidates = held;
        let mut heaped = Vec::new();
        while place(&candidates, 9).is_none() {
            heaped.push(candidates.remove(spill_first(&candidates)).var);
        }
        assert_eq!(heaped, vec![ValueId::from_raw(0)]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].depth, 1, "the loop keeps its aggregate");
    }
}
