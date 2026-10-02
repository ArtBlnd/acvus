//! The IR printer: a body or a module as text, in notation alone. It is
//! never read back.
//!
//! What it prints is a `View`, a set of items each on or off; `View::CFG`
//! and `View::FULL` are two of them.
//! - Always: each block by its parameters, instructions and terminator; each
//!   value with its type (`ty::print`); the body's entry.
//! - `header`: the body's parameters, captures, order parameter and slots,
//!   each with its name and type, and its result type.
//! - `spans`: each instruction's and slot's span, `@start..end`.
//! - `order`: the order tokens each instruction consumes and yields.
//! - `structure`: a `Diamond`'s and a `Switch`'s join.
//! - `origins`: each value's `ValOrigin`.
//! - `canonical`: values, blocks and instructions renumbered in the order
//!   they print, blocks in reverse postorder from the entry, blocks no path
//!   reaches after them by their ids; off, each prints by its own id.
//!
//! Annotations are what an analysis says at a point, given by `Annotate`
//! and printed on `#` lines above the instruction, the block or the edge
//! they are at, each `key: value`, keys in order. An annotation writes each
//! value and block through the print's `Names`, so it names them as the
//! print does, canonical or not. The printer knows no analysis.
//!
//! Notation only: an instruction prints as its operation's name and its
//! operands, a place as its root and path (`slot0.a`, `*v3.b`, `v4[v5]`), a
//! name by the interner, a type by `ty::print`. Nothing else is written.
//!
//! Deterministic: the same module and view print the same text. Nothing is
//! printed in a hash map's order; every set and map is printed sorted.

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, LocalIdOps, QualifiedRef};
use rustc_hash::FxHashMap;

use super::{Block, BlockId, Body, BodyId, Definition, InstId, Module, Op, OrderUse, Terminator};
use crate::decl::PairMember;
use crate::ir::{BinOp, Callee, ForSource, IndexBound, IndexMode, Intrinsic, PathSeg, RefTarget, SwitchKey, UnaryOp, ValOrigin, ValueId};
use crate::ty::{self, Mutability, Ty};

/// The items a print shows beyond blocks, instructions, terminators and
/// types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub header: bool,
    pub spans: bool,
    pub order: bool,
    pub structure: bool,
    pub origins: bool,
    pub canonical: bool,
}

impl View {
    /// The control flow and the types.
    pub const CFG: View = View { header: false, spans: false, order: false, structure: false, origins: false, canonical: false };
    /// Everything the body holds.
    pub const FULL: View = View { header: true, spans: true, order: true, structure: true, origins: true, canonical: false };
}

/// What an analysis says at points of one body, as `key: value` pairs in
/// notation, each value and block written through `names`. A point it says
/// nothing of gives none.
pub trait Annotate {
    /// At a block's start.
    fn block(&self, names: &Names<'_>, block: BlockId) -> Vec<(&'static str, String)>;
    /// Just before an instruction.
    fn before(&self, names: &Names<'_>, inst: InstId) -> Vec<(&'static str, String)>;
    /// On an edge, from the block whose terminator takes it.
    fn edge(&self, names: &Names<'_>, from: BlockId, to: BlockId) -> Vec<(&'static str, String)>;
}

/// `body` of `module` as `view` shows it, with each of `annotations`.
pub fn body(module: &Module, id: BodyId, names: &Interner, view: View, annotations: &[&dyn Annotate]) -> String {
    let body = module.body(id);
    let ids = Names { body, numbering: Numbering::of(body, view.canonical) };
    let printer = Printer { module, body, names, view, ids: &ids, annotations };
    let mut out = String::new();
    printer.body(id, &mut out);
    out
}

/// Every body of `module`, by `BodyId`, as `view` shows it. Annotations are
/// of one body each, so a module prints none.
pub fn module(module: &Module, names: &Interner, view: View) -> String {
    module
        .body_ids()
        .map(|id| body(module, id, names, view, &[]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A path under a place, `.a.0`.
pub(crate) fn path(names: &Interner, path: &[PathSeg]) -> String {
    path.iter()
        .map(|seg| match seg {
            PathSeg::Field(field) => format!(".{}", names.resolve(*field)),
            PathSeg::Index(index) => format!(".{index}"),
            PathSeg::Payload => ".Payload".to_string(),
        })
        .collect()
}

/// `@start..end`.
pub(crate) fn span(span: Span) -> String {
    format!("@{}..{}", span.start, span.end)
}

/// The names one print gives the values and blocks of its body: a value by
/// its role, `p0`, `c0`, `slot0`, a token `t3`, any other value `v3`, and a
/// block `b2`, each numbered as the print's `canonical` says.
pub struct Names<'b> {
    body: &'b Body,
    numbering: Numbering,
}

impl Names<'_> {
    pub fn value(&self, value: ValueId) -> String {
        self.numbering.value(self.body, value)
    }

    pub fn block(&self, block: BlockId) -> String {
        self.numbering.block(block)
    }

    /// Sorts `values` in the order their names print: parameters,
    /// captures, slots, tokens, then every other value, each kind by its
    /// number.
    pub fn sort_values(&self, values: &mut [ValueId]) {
        values.sort_by_key(|value| self.numbering.value_key(self.body, *value));
    }

    fn values(&self, values: &[ValueId]) -> String {
        self.numbering.values(self.body, values)
    }

    fn place(&self, names: &Interner, target: &RefTarget, at: &[PathSeg]) -> String {
        self.numbering.place(self.body, names, target, at)
    }

    fn blocks(&self) -> Vec<BlockId> {
        self.numbering.blocks(self.body)
    }
}

/// Where a value's name sorts among others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ValueKey {
    Param(usize),
    Capture(usize),
    Slot(usize),
    Token(usize),
    Value(usize),
}

/// How values and blocks are numbered in one print.
enum Numbering {
    /// Each by its own id; blocks in id order.
    Own,
    /// Each in the order it prints; blocks in reverse postorder from the
    /// entry, then those no path reaches, by id.
    Canonical {
        blocks: Vec<BlockId>,
        block_numbers: FxHashMap<BlockId, usize>,
        value_numbers: FxHashMap<ValueId, usize>,
    },
}

impl Numbering {
    fn of(body: &Body, canonical: bool) -> Numbering {
        if !canonical {
            return Numbering::Own;
        }
        let blocks = canonical_block_order(body);
        let block_numbers = blocks.iter().enumerate().map(|(number, block)| (*block, number)).collect();
        let mut value_numbers = FxHashMap::default();
        let mut tokens = 0;
        let mut values = 0;
        let mut number = |value: ValueId| {
            let counter = if body.is_token(value) { &mut tokens } else { &mut values };
            value_numbers.insert(value, *counter);
            *counter += 1;
        };
        if let Some(order) = body.order_param() {
            number(order);
        }
        for block in &blocks {
            let block = body.block(*block);
            block.params().iter().copied().for_each(&mut number);
            for inst in block.insts() {
                let yielded = inst.order().and_then(OrderUse::yielded);
                yielded.into_iter().chain(inst.result()).for_each(&mut number);
            }
        }
        Numbering::Canonical { blocks, block_numbers, value_numbers }
    }

    fn blocks(&self, body: &Body) -> Vec<BlockId> {
        match self {
            Numbering::Own => body.blocks().map(|(id, _)| id).collect(),
            Numbering::Canonical { blocks, .. } => blocks.clone(),
        }
    }

    fn block(&self, block: BlockId) -> String {
        match self {
            Numbering::Own => format!("b{}", block.index()),
            Numbering::Canonical { block_numbers, .. } => {
                let number = block_numbers.get(&block).unwrap_or_else(|| panic!("{block:?} is a block of the body printed"));
                format!("b{number}")
            }
        }
    }

    fn value(&self, body: &Body, value: ValueId) -> String {
        match self.value_key(body, value) {
            ValueKey::Param(index) => format!("p{index}"),
            ValueKey::Capture(index) => format!("c{index}"),
            ValueKey::Slot(index) => format!("slot{index}"),
            ValueKey::Token(number) => format!("t{number}"),
            ValueKey::Value(number) => format!("v{number}"),
        }
    }

    fn value_key(&self, body: &Body, value: ValueId) -> ValueKey {
        match body.definition(value) {
            Definition::Param(index) => ValueKey::Param(index),
            Definition::Capture(index) => ValueKey::Capture(index),
            Definition::Slot(index) => ValueKey::Slot(index),
            Definition::OrderParam | Definition::BlockParam(..) | Definition::Inst(_) => {
                let number = match self {
                    Numbering::Own => value.to_raw(),
                    Numbering::Canonical { value_numbers, .. } => *value_numbers
                        .get(&value)
                        .unwrap_or_else(|| panic!("{value:?} is defined in the body printed")),
                };
                if body.is_token(value) { ValueKey::Token(number) } else { ValueKey::Value(number) }
            }
        }
    }

    fn values(&self, body: &Body, values: &[ValueId]) -> String {
        values.iter().map(|v| self.value(body, *v)).collect::<Vec<_>>().join(", ")
    }

    fn place(&self, body: &Body, names: &Interner, target: &RefTarget, at: &[PathSeg]) -> String {
        let root = match target {
            RefTarget::Var(storage) | RefTarget::Param(storage) => self.value(body, *storage),
            RefTarget::Through(reference) => format!("*{}", self.value(body, *reference)),
            RefTarget::Context(context) => format!("@{}", ty::print::qualified(names, *context)),
        };
        format!("{root}{}", path(names, at))
    }
}

/// The blocks in reverse postorder of a depth-first walk of the edges from
/// the entry, each block's edges taken in its terminator's order, then the
/// blocks the walk does not reach, by id.
fn canonical_block_order(body: &Body) -> Vec<BlockId> {
    let count = body.block_count();
    let mut visited = vec![false; count];
    let mut postorder = Vec::with_capacity(count);
    let mut path: Vec<(BlockId, usize)> = vec![(body.entry(), 0)];
    visited[body.entry().index()] = true;
    while let Some((block, next)) = path.last_mut() {
        let edges = body.block(*block).terminator().edges();
        let Some(edge) = edges.get(*next) else {
            postorder.push(*block);
            path.pop();
            continue;
        };
        *next += 1;
        let to = edge.to;
        if !visited[to.index()] {
            visited[to.index()] = true;
            path.push((to, 0));
        }
    }
    postorder.reverse();
    let unreached = body.blocks().map(|(id, _)| id).filter(|id| !visited[id.index()]);
    postorder.into_iter().chain(unreached).collect()
}

struct Printer<'p> {
    module: &'p Module,
    body: &'p Body,
    names: &'p Interner,
    view: View,
    ids: &'p Names<'p>,
    annotations: &'p [&'p dyn Annotate],
}

impl Printer<'_> {
    fn body(&self, id: BodyId, out: &mut String) {
        let mut first = format!("body{}", id.0);
        if self.view.header {
            first.push_str(&self.signature());
        }
        first.push_str(&format!(" entry={}", self.block_name(self.body.entry())));
        line(out, 0, &first);
        if self.view.header {
            for slot in self.body.slots() {
                let mut text = self.definition(*slot, self.slot_name(*slot));
                if self.view.spans {
                    text.push_str(&format!(" {}", span(self.body.slot_spans[slot])));
                }
                line(out, 0, &text);
            }
        }
        for block in self.ids.blocks() {
            self.block(block, out);
        }
    }

    /// ` (p0 n: i64) cap(c0 f: T) order=t0 -> R`.
    fn signature(&self) -> String {
        let named = |values: &[(acvus_utils::Astr, ValueId)]| {
            values
                .iter()
                .map(|(name, value)| self.definition(*value, Some(self.names.resolve(*name).to_string())))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut text = format!(" ({})", named(self.body.params()));
        if !self.body.captures().is_empty() {
            text.push_str(&format!(" cap({})", named(self.body.captures())));
        }
        if let Some(order) = self.body.order_param() {
            text.push_str(&format!(" order={}", self.value(order)));
        }
        text.push_str(&format!(" -> {}", self.ty(self.body.ret())));
        text
    }

    /// A slot's name: its origin, where it has one.
    fn slot_name(&self, slot: ValueId) -> Option<String> {
        self.body.debug().val_origins.get(&slot).map(|origin| self.origin(origin))
    }

    fn block(&self, id: BlockId, out: &mut String) {
        let block = self.body.block(id);
        self.annotate(out, 0, |annotation| annotation.block(self.ids, id));
        let params: Vec<String> = block.params().iter().map(|param| self.definition(*param, self.origin_of(*param))).collect();
        line(out, 0, &format!("{}({}):", self.block_name(id), params.join(", ")));
        for inst in block.insts() {
            self.annotate(out, 1, |annotation| annotation.before(self.ids, inst.id()));
            line(out, 1, &self.inst(inst));
        }
        self.edge_annotations(id, block, out);
        let mut terminator = self.terminator(block.terminator());
        if self.view.spans {
            terminator.push_str(&format!(" {}", span(block.terminator_span())));
        }
        line(out, 1, &terminator);
    }

    fn edge_annotations(&self, from: BlockId, block: Block<'_>, out: &mut String) {
        let mut targets: Vec<BlockId> = Vec::new();
        for edge in block.terminator().edges() {
            if !targets.contains(&edge.to) {
                targets.push(edge.to);
            }
        }
        for to in targets {
            let prefix = format!("-> {} ", self.block_name(to));
            let mut pairs: Vec<(&'static str, String)> =
                self.annotations.iter().flat_map(|annotation| annotation.edge(self.ids, from, to)).collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            for (key, value) in pairs {
                line(out, 1, &format!("# {prefix}{key}: {value}"));
            }
        }
    }

    fn annotate<F>(&self, out: &mut String, depth: usize, at: F)
    where
        F: Fn(&dyn Annotate) -> Vec<(&'static str, String)>,
    {
        let mut pairs: Vec<(&'static str, String)> = self.annotations.iter().flat_map(|annotation| at(*annotation)).collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        for (key, value) in pairs {
            line(out, depth, &format!("# {key}: {value}"));
        }
    }

    fn inst(&self, inst: &super::Inst) -> String {
        let mut text = String::new();
        let order = inst.order().filter(|_| self.view.order);
        let yielded = order.and_then(OrderUse::yielded);
        if let Some(token) = yielded {
            text.push_str(&self.value(token));
        }
        if let Some(result) = inst.result() {
            if yielded.is_some() {
                text.push_str(", ");
            }
            text.push_str(&self.definition(result, self.origin_of(result)));
        }
        if !text.is_empty() {
            text.push_str(" = ");
        }
        text.push_str(&self.op(inst.op()));
        if let Some(token) = order.and_then(OrderUse::consumed) {
            text.push_str(&format!(" {}", self.value(token)));
        }
        if self.view.spans {
            text.push_str(&format!(" {}", span(inst.span())));
        }
        text
    }

    /// `v3 x: i64`: a value, its origin where it is shown, and its type.
    fn definition(&self, value: ValueId, name: Option<String>) -> String {
        let mut text = self.value(value);
        if let Some(name) = name {
            text.push(' ');
            text.push_str(&name);
        }
        text.push_str(": ");
        text.push_str(&self.ty(self.body.ty_of(value)));
        text
    }

    fn origin_of(&self, value: ValueId) -> Option<String> {
        if !self.view.origins {
            return None;
        }
        self.body.debug().val_origins.get(&value).map(|origin| self.origin(origin))
    }

    fn origin(&self, origin: &ValOrigin) -> String {
        match origin {
            ValOrigin::Named(name) => self.names.resolve(*name).to_string(),
            ValOrigin::Context(name) => format!("@{}", self.names.resolve(*name)),
            ValOrigin::ExternParam(name) => format!("${}", self.names.resolve(*name)),
            ValOrigin::Field(object, field) => format!("{}.{}", self.value(*object), self.names.resolve(*field)),
            ValOrigin::RefField(target, at) => self.place(target, at),
            ValOrigin::Call(name) => format!("{}()", self.names.resolve(*name)),
            ValOrigin::Expr => "_".to_string(),
        }
    }

    fn op(&self, op: &Op) -> String {
        match op {
            Op::Const { value } => format!("Const {}", literal(value)),
            Op::ConstStr { text } => format!("ConstStr {text:?}"),
            Op::StringConcat { parts } => format!("StringConcat {}", self.values(parts)),
            Op::StringAppend { target, part } => format!("StringAppend {}", self.values(&[*target, *part])),
            Op::StringEq { a, b } => format!("StringEq {}", self.values(&[*a, *b])),
            Op::StringClone { src } => format!("StringClone {}", self.value(*src)),
            Op::Ref { target, path, mutability } => {
                format!("Ref{} {}", mutability_flag(*mutability), self.place(target, path))
            }
            Op::Take { target, path, taken_out } => {
                format!("Take{} {}", flag("taken_out", *taken_out), self.place(target, path))
            }
            Op::Assign { target, path, value, restores } => format!(
                "Assign{} {}, {}",
                flag("restores", *restores),
                self.place(target, path),
                self.value(*value)
            ),
            Op::AsSlice { container, mutability, r#impl } => format!(
                "AsSlice{} {} {}",
                mutability_flag(*mutability),
                self.qualified(*r#impl),
                self.value(*container)
            ),
            Op::ArrayView { src, mutability } => format!("ArrayView{} {}", mutability_flag(*mutability), self.value(*src)),
            Op::SliceLen { src } => format!("SliceLen {}", self.value(*src)),
            Op::Index { slice, index, mode, bound } => format!(
                "Index({}, {}) {}[{}]",
                index_mode(*mode),
                index_bound(*bound),
                self.value(*slice),
                self.value(*index)
            ),
            Op::IndexSet { slice, index, value, bound } => format!(
                "IndexSet({}) {}[{}], {}",
                index_bound(*bound),
                self.value(*slice),
                self.value(*index),
                self.value(*value)
            ),
            Op::FieldGet { object, field, rest } => format!("FieldGet {}{}", self.value(*object), self.fields(*field, rest)),
            Op::FieldSet { object, field, rest, value } => format!(
                "FieldSet {}{}, {}",
                self.value(*object),
                self.fields(*field, rest),
                self.value(*value)
            ),
            Op::BinOp { op, left, right } => format!("{} {}", bin_op(*op), self.values(&[*left, *right])),
            Op::UnaryOp { op, operand } => format!("{} {}", unary_op(*op), self.value(*operand)),
            Op::Cast { src, to } => format!("Cast({}) {}", to.name(), self.value(*src)),
            Op::Unmark { src } => format!("Unmark {}", self.value(*src)),
            Op::FunctionCall { callee, callee_ty: _, args } => format!("Call {}({})", self.callee(callee), self.values(args)),
            Op::Spawn { callee, callee_ty: _, args } => format!("Spawn {}({})", self.callee(callee), self.values(args)),
            Op::TakeEffect { src } => format!("TakeEffect {}", self.value(*src)),
            Op::Merge { orders } => format!("Merge {}", self.values(orders)),
            Op::ArrayBegin { capacity } => format!("ArrayBegin({capacity})"),
            Op::ArrayPush { array, value } => format!("ArrayPush {}", self.values(&[*array, *value])),
            Op::MakeObject { fields } => {
                let fields: Vec<String> = fields
                    .iter()
                    .map(|(name, value)| format!("{}: {}", self.names.resolve(*name), self.value(*value)))
                    .collect();
                format!("MakeObject {{{}}}", fields.join(", "))
            }
            Op::Intrinsic { intrinsic, args } => format!("Intrinsic({}) {}", intrinsic_name(*intrinsic), self.values(args)),
            Op::MakeTuple { elements } => format!("MakeTuple {}", self.values(elements)),
            Op::TupleIndex { tuple, index } => format!("TupleIndex {}.{index}", self.value(*tuple)),
            Op::TestLiteral { src, value } => format!("TestLiteral {}, {}", self.value(*src), literal(value)),
            Op::TestObjectKey { src, key } => format!("TestObjectKey {}.{}", self.value(*src), self.names.resolve(*key)),
            Op::ArrayIndex { array, index } => format!("ArrayIndex {}[{index}]", self.value(*array)),
            Op::ObjectGet { object, key } => format!("ObjectGet {}.{}", self.value(*object), self.names.resolve(*key)),
            Op::MakeClosure { bodies, captures } => {
                let bodies: Vec<String> = bodies.iter().map(|body| format!("body{}", body.0)).collect();
                format!("MakeClosure [{}]({})", bodies.join(", "), self.values(captures))
            }
            Op::Specialize { closure, signatures } => {
                let signatures: Vec<String> = signatures.iter().map(|signature| self.ty(*signature)).collect();
                format!("Specialize {} [{}]", self.value(*closure), signatures.join(", "))
            }
            Op::MakeVariant { tag, payload } => match payload {
                Some(payload) => format!("MakeVariant {}({})", self.names.resolve(*tag), self.value(*payload)),
                None => format!("MakeVariant {}", self.names.resolve(*tag)),
            },
            Op::TestVariant { src, tag } => format!("TestVariant {}, {}", self.value(*src), self.names.resolve(*tag)),
            Op::UnwrapVariant { src } => format!("UnwrapVariant {}", self.value(*src)),
            Op::Undef => "Undef".to_string(),
            Op::Drop { src } => format!("Drop {}", self.value(*src)),
        }
    }

    fn terminator(&self, terminator: &Terminator) -> String {
        match terminator {
            Terminator::Jump { to, args } => format!("Jump {}", self.target(*to, args)),
            Terminator::JumpIf { cond, then_to, then_args, else_to, else_args } => format!(
                "JumpIf {} ? {} : {}",
                self.value(*cond),
                self.target(*then_to, then_args),
                self.target(*else_to, else_args)
            ),
            Terminator::Diamond { cond, then_to, then_args, else_to, else_args, join } => {
                let mut text = format!(
                    "Diamond {} ? {} : {}",
                    self.value(*cond),
                    self.target(*then_to, then_args),
                    self.target(*else_to, else_args)
                );
                if self.view.structure {
                    text.push_str(&format!(" join {}", self.block_name(*join)));
                }
                text
            }
            Terminator::Switch { tag, arms, default, join } => {
                let mut cases: Vec<String> = arms
                    .iter()
                    .map(|(key, to, args)| format!("{} => {}", self.switch_key(*key), self.target(*to, args)))
                    .collect();
                if let Some((to, args)) = default {
                    cases.push(format!("_ => {}", self.target(*to, args)));
                }
                let mut text = format!("Switch {} {{{}}}", self.value(*tag), cases.join(", "));
                if self.view.structure {
                    text.push_str(&format!(" join {}", self.block_name(*join)));
                }
                text
            }
            Terminator::For { source, body, body_args, exit, exit_args } => {
                format!("For {} ? {} : {}", self.for_source(source), self.target(*body, body_args), self.target(*exit, exit_args))
            }
            Terminator::While { cond, body, body_args, exit, exit_args } => {
                format!("While {} ? {} : {}", self.value(*cond), self.target(*body, body_args), self.target(*exit, exit_args))
            }
            Terminator::Return { value, order, span: _ } => {
                let mut text = format!("Return {}", self.value(*value));
                if let Some(token) = order.filter(|_| self.view.order) {
                    text.push_str(&format!(" {}", self.value(token)));
                }
                text
            }
            Terminator::Diverge => "Diverge".to_string(),
            Terminator::NoArmMatched { span: _ } => "NoArmMatched".to_string(),
        }
    }

    fn for_source(&self, source: &ForSource) -> String {
        match source {
            ForSource::Slice(value) => format!("Slice({})", self.value(*value)),
            ForSource::SliceMut(value) => format!("SliceMut({})", self.value(*value)),
            ForSource::Array(value) => format!("Array({})", self.value(*value)),
            ForSource::Range { at, hi } => format!("Range({}, {})", self.value(*at), self.value(*hi)),
        }
    }

    fn switch_key(&self, key: SwitchKey) -> String {
        match key {
            SwitchKey::Tag(tag) => self.names.resolve(tag).to_string(),
            SwitchKey::Int(n) => n.to_string(),
            SwitchKey::Bool(b) => b.to_string(),
            SwitchKey::Char(c) => format!("{c:?}"),
            SwitchKey::Str(s) => format!("{:?}", self.names.resolve(s)),
        }
    }

    fn target(&self, to: BlockId, args: &[ValueId]) -> String {
        format!("{}({})", self.block_name(to), self.values(args))
    }

    fn callee(&self, callee: &Callee) -> String {
        match callee {
            Callee::Direct(id) => self.qualified(*id),
            Callee::Extern { id, required: _, member } => match member {
                PairMember::Uniform => self.qualified(*id),
                PairMember::Specialized => format!("#{}", self.qualified(*id)),
            },
            Callee::Indirect(value) => self.value(*value),
        }
    }

    /// `.a.b`: a field and the fields below it.
    fn fields(&self, field: acvus_utils::Astr, rest: &[acvus_utils::Astr]) -> String {
        std::iter::once(field).chain(rest.iter().copied()).map(|name| format!(".{}", self.names.resolve(name))).collect()
    }

    fn value(&self, value: ValueId) -> String {
        self.ids.value(value)
    }

    fn values(&self, values: &[ValueId]) -> String {
        self.ids.values(values)
    }

    fn place(&self, target: &RefTarget, at: &[PathSeg]) -> String {
        self.ids.place(self.names, target, at)
    }

    fn block_name(&self, block: BlockId) -> String {
        self.ids.block(block)
    }

    fn ty(&self, ty: Ty) -> String {
        ty::print::ty(self.module.types(), self.names, ty)
    }

    fn qualified(&self, name: QualifiedRef) -> String {
        ty::print::qualified(self.names, name)
    }
}

fn line(out: &mut String, depth: usize, text: &str) {
    for _ in 0..depth {
        out.push_str("  ");
    }
    out.push_str(text);
    out.push('\n');
}

/// `(mut)` for an exclusive reference; a shared one, as `&`, carries no
/// mark.
fn mutability_flag(mutability: Mutability) -> &'static str {
    match mutability {
        Mutability::Mut => "(mut)",
        Mutability::Shared => "",
    }
}

/// `(name)` where the flag is set, nothing where it is not.
fn flag(name: &str, set: bool) -> String {
    if set { format!("({name})") } else { String::new() }
}

fn literal(value: &Literal) -> String {
    match value {
        Literal::String(text) => format!("{text:?}"),
        Literal::Int(n) => n.to_string(),
        Literal::IntOf(suffixed) => format!("{}{}", suffixed.value, suffixed.width.name()),
        Literal::Float(x) => format!("{x:?}"),
        Literal::Char(c) => format!("{c:?}"),
        Literal::Bytes(bytes) => format!("b\"{}\"", bytes.escape_ascii()),
        Literal::Bool(b) => b.to_string(),
        Literal::List(items) => format!("[{}]", items.iter().map(literal).collect::<Vec<_>>().join(", ")),
        Literal::Unit => "()".to_string(),
    }
}

fn bin_op(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "Add",
        BinOp::Sub => "Sub",
        BinOp::Mul => "Mul",
        BinOp::Shl => "Shl",
        BinOp::Shr => "Shr",
        BinOp::Div => "Div",
        BinOp::Eq => "Eq",
        BinOp::Neq => "Neq",
        BinOp::Lt => "Lt",
        BinOp::Gt => "Gt",
        BinOp::Lte => "Lte",
        BinOp::Gte => "Gte",
        BinOp::And => "And",
        BinOp::Or => "Or",
        BinOp::Xor => "Xor",
        BinOp::BitAnd => "BitAnd",
        BinOp::BitOr => "BitOr",
        BinOp::Mod => "Mod",
        BinOp::Min => "Min",
        BinOp::Max => "Max",
    }
}

fn unary_op(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "Neg",
        UnaryOp::Not => "Not",
    }
}

fn index_mode(mode: IndexMode) -> &'static str {
    match mode {
        IndexMode::Copy => "Copy",
        IndexMode::Ref => "Ref",
    }
}

fn index_bound(bound: IndexBound) -> &'static str {
    match bound {
        IndexBound::Checked => "Checked",
        IndexBound::Proven => "Proven",
    }
}

fn intrinsic_name(intrinsic: Intrinsic) -> &'static str {
    match intrinsic {
        Intrinsic::StringClone => "StringClone",
        Intrinsic::Gather => "Gather",
        Intrinsic::ArrayMap => "ArrayMap",
    }
}

#[cfg(test)]
mod tests;
