use std::fmt;

use acvus_ast::Literal;

use crate::ir::{BinOp, Overflow, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::analysis::cost::{CostTable, Costs, InPlace, LoopCost};
use crate::analysis::loop_deps::{
    Accumulator, BodyDeps, CallIdentity, Control, Cycle, Law, LawOp, LoopDeps, Member,
    Order, Placement, Storage, Token,
};
use crate::analysis::loops::{Term, Trip};
use crate::cfg::{CfgBody, Terminator, promote};
use crate::ir::{
    Callee, ExitTrip, ForSource, IndexBound, IndexMode, InstKind, Label, MirBody, MirModule,
    ValueId,
};
use crate::laws::LawTable;

/// Normalizes ValueIds to sequential order of first appearance.
struct ValNormalizer {
    map: FxHashMap<ValueId, usize>,
}

impl ValNormalizer {
    fn new() -> Self {
        Self {
            map: FxHashMap::default(),
        }
    }

    fn get(&mut self, v: ValueId) -> usize {
        let len = self.map.len();
        *self.map.entry(v).or_insert(len)
    }

    fn fmt_val(&mut self, r: ValueId) -> String {
        format!("r{}", self.get(r))
    }

    fn fmt_use(
        &mut self,
        r: ValueId,
        consts: &FxHashMap<ValueId, &Literal>,
        texts: &FxHashMap<ValueId, usize>,
    ) -> String {
        if let Some(tidx) = texts.get(&r) {
            return format!("T{tidx}");
        }
        match consts.get(&r) {
            Some(lit) => format!("{} ({})", fmt_literal(lit), self.fmt_val(r)),
            None => self.fmt_val(r),
        }
    }

    fn fmt_uses(
        &mut self,
        regs: &[ValueId],
        consts: &FxHashMap<ValueId, &Literal>,
        texts: &FxHashMap<ValueId, usize>,
    ) -> String {
        regs.iter()
            .map(|r| self.fmt_use(*r, consts, texts))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn proven_suffix(bound: IndexBound) -> &'static str {
    match bound {
        IndexBound::Checked => "",
        IndexBound::Proven => " proven",
    }
}

/// `Op(Add) exact commutative`, `Call(#0, option-lifted) exact`,
/// `Fold(r3, #1) exact`, `Order exact commutative`.
fn fmt_accumulator(acc: &Accumulator, ctx: &PrintCtx<'_>, vn: &mut ValNormalizer) -> String {
    let law = match &acc.law {
        Law::Op(LawOp::Add) => "Op(Add)".to_string(),
        Law::Op(LawOp::Mul) => "Op(Mul)".to_string(),
        Law::Op(LawOp::Min) => "Op(Min)".to_string(),
        Law::Op(LawOp::Max) => "Op(Max)".to_string(),
        Law::Op(LawOp::Concat) => "Op(Concat)".to_string(),
        Law::Call(call) => {
            let identity = match call.identity {
                CallIdentity::Declared(_) => "identity",
                CallIdentity::OptionLifted => "option-lifted",
            };
            format!("Call({}, {identity})", ctx.fmt_fn_id(call.callee.id))
        }
        Law::Fold(fold) => format!(
            "Fold({}, {})",
            vn.fmt_val(fold.storage),
            ctx.fmt_fn_id(fold.callee.id)
        ),
        Law::Order => "Order".to_string(),
    };
    let exact = if acc.exact { " exact" } else { " inexact" };
    let commutative = if acc.commutative { " commutative" } else { "" };
    format!("{law}{exact}{commutative}")
}

/// What the listing shows of each `For`'s stages: the boundaries the
/// terminator states, or those and what `analysis::loop_deps` computes of
/// each stage, which needs the law table, and with a backend's table what
/// `analysis::cost` computes of the loop.
#[derive(Clone, Copy)]
pub enum StageFacts<'a> {
    Boundaries,
    Computed {
        laws: &'a LawTable,
        costs: Option<&'a CostTable>,
    },
}

/// The comment lines under one `For`: per stage, `free` or its cycles, each
/// with its tokens, order and law and the operations it holds; a cycle that
/// crosses a boundary; and the loop's control.
fn fmt_loop_facts(
    deps: &LoopDeps,
    cfg: &CfgBody,
    laws: &LawTable,
    ctx: &PrintCtx<'_>,
    vn: &mut ValNormalizer,
) -> Vec<String> {
    let judged = deps.judge(cfg, laws);
    let mut lines = Vec::new();
    for (stage, blocks) in deps.membership.stages().iter().enumerate() {
        let entry = fmt_label(blocks.entry);
        if deps.is_free(stage) {
            let held: Vec<String> = blocks
                .blocks
                .iter()
                .flat_map(|block| &cfg.blocks[block.0].insts)
                .map(|inst| mnemonic(&inst.kind, ctx))
                .collect();
            lines.push(format!("{entry}: free {{{}}}", held.join(", ")));
            continue;
        }
        for (cycle, judged) in deps.cycles.iter().zip(&judged) {
            if cycle.stage() == Some(stage) {
                lines.push(format!(
                    "{entry}: {}",
                    fmt_cycle(cycle, judged.order, judged.law.as_ref(), cfg, ctx, vn)
                ));
            }
        }
    }
    for (cycle, judged) in deps.cycles.iter().zip(&judged) {
        let Placement::Crosses(stages) = &cycle.placement else {
            continue;
        };
        let entries: Vec<String> = stages
            .iter()
            .map(|stage| fmt_label(deps.membership.stages()[*stage].entry))
            .collect();
        lines.push(format!(
            "crosses {}: {}",
            entries.join(", "),
            fmt_cycle(cycle, judged.order, judged.law.as_ref(), cfg, ctx, vn)
        ));
    }
    lines.push(match deps.control {
        Control::Upfront => "control upfront".to_string(),
        Control::Chained { cycle } => match deps.cycles[cycle].stage() {
            Some(stage) => format!(
                "control chained through {}",
                fmt_label(deps.membership.stages()[stage].entry)
            ),
            None => "control chained across stages".to_string(),
        },
    });
    lines
}

/// `cost W=4 O=4096 split when n > 32768, n = len(r3)` (RFC-0066 rule 8),
/// or `cost in place: no free stage`.
fn fmt_cost(cost: LoopCost, trip: Option<String>) -> String {
    match cost {
        LoopCost::Split {
            work,
            overhead,
            threshold,
        } => {
            let named = match trip {
                Some(term) => format!(", n = {term}"),
                None => String::new(),
            };
            format!("cost W={work} O={overhead} split when n > {threshold}{named}")
        }
        LoopCost::InPlace(InPlace::NoFreeStage) => "cost in place: no free stage".to_string(),
        LoopCost::InPlace(InPlace::NoWork) => "cost in place: W=0".to_string(),
    }
}

/// A trip count as RFC-0066 rule 3 writes it: `max(r7 - 0 (r3), 0)`.
fn fmt_term(
    term: &Term,
    vn: &mut ValNormalizer,
    consts: &FxHashMap<ValueId, &Literal>,
    texts: &FxHashMap<ValueId, usize>,
) -> String {
    let operand = |term: &Term, vn: &mut ValNormalizer| match term {
        Term::Add(..) | Term::Sub(..) | Term::Mul(..) => {
            format!("({})", fmt_term(term, vn, consts, texts))
        }
        Term::Const(_) | Term::Value(_) | Term::Len(_) | Term::Max(..) => {
            fmt_term(term, vn, consts, texts)
        }
    };
    match term {
        Term::Const(literal) => fmt_literal(literal),
        Term::Value(value) => vn.fmt_use(*value, consts, texts),
        Term::Len(source) => format!("len({})", vn.fmt_use(*source, consts, texts)),
        Term::Add(a, b) => format!("{} + {}", operand(a, vn), operand(b, vn)),
        Term::Sub(a, b) => format!("{} - {}", operand(a, vn), operand(b, vn)),
        Term::Mul(a, b) => format!("{} * {}", operand(a, vn), operand(b, vn)),
        Term::Max(a, b) => format!(
            "max({}, {})",
            fmt_term(a, vn, consts, texts),
            fmt_term(b, vn, consts, texts)
        ),
    }
}

/// `cycle Carried(r3) any_order law(Op(Add) exact commutative) {+}`.
fn fmt_cycle(
    cycle: &Cycle,
    order: Order,
    law: Option<&Accumulator>,
    cfg: &CfgBody,
    ctx: &PrintCtx<'_>,
    vn: &mut ValNormalizer,
) -> String {
    let tokens: Vec<String> = cycle
        .tokens
        .iter()
        .map(|token| match token {
            Token::Order(value) => format!("Order({})", vn.fmt_val(*value)),
            Token::Carried(value) => format!("Carried({})", vn.fmt_val(*value)),
            Token::Storage(Storage::Slot(slot)) => format!("Storage({})", vn.fmt_val(*slot)),
            Token::Storage(Storage::Context(context)) => {
                format!("Storage(@{})", ctx.interner.resolve(context.name))
            }
            Token::Storage(Storage::Element) => "Storage(element)".to_string(),
            Token::Control => "Control".to_string(),
        })
        .collect();
    let order = match order {
        Order::Disjoint => "disjoint",
        Order::AnyOrder => "any_order",
        Order::InOrder => "in_order",
    };
    let law = match law {
        Some(acc) => format!(" law({})", fmt_accumulator(acc, ctx, vn)),
        None => String::new(),
    };
    let held: Vec<String> = cycle
        .members
        .iter()
        .filter_map(|member| match *member {
            Member::Inst(at) => Some(mnemonic(&cfg.blocks[at.block.0].insts[at.at].kind, ctx)),
            Member::Term(block) => terminator_mnemonic(&cfg.blocks[block.0].terminator),
        })
        .collect();
    format!(
        "cycle {} {order}{law} {{{}}}",
        tokens.join("+"),
        held.join(", ")
    )
}

/// A terminator as a cycle names it; a plain jump passing nothing is only
/// the edge between two blocks, and is not named.
fn terminator_mnemonic(term: &Terminator) -> Option<String> {
    let name = match term {
        Terminator::Jump { args, .. } if args.is_empty() => return None,
        Terminator::Jump { .. } => "jump",
        Terminator::JumpIf { .. } | Terminator::Diamond { .. } => "branch",
        Terminator::Switch { .. } => "switch",
        Terminator::For { .. } => "for",
        Terminator::Return { .. } => "return",
        Terminator::Diverge => "diverge",
        Terminator::Fallthrough => "fallthrough",
    };
    Some(name.to_string())
}

/// An instruction as a stage's summary names it: an operator by its
/// spelling, a call by its callee's name, anything else by its kind.
fn mnemonic(kind: &InstKind, ctx: &PrintCtx<'_>) -> String {
    let callee_name = |callee: &Callee| match callee {
        Callee::Direct(id) | Callee::Extern { id, .. } => {
            ctx.interner.resolve(id.name).to_string()
        }
        Callee::Indirect(_) => "indirect".to_string(),
    };
    let name = match kind {
        InstKind::BinOp { op, .. } => {
            return match fmt_binop(*op) {
                Spelling::Infix(spelled) | Spelling::Call(spelled) => spelled.to_string(),
            };
        }
        InstKind::UnaryOp { op, .. } => return fmt_unaryop(*op).to_string(),
        InstKind::Check { .. } => "check",
        InstKind::CheckSteps { .. } => "check_steps",
        InstKind::FunctionCall { callee, .. } => return format!("call {}", callee_name(callee)),
        InstKind::Spawn { callee, .. } => return format!("spawn {}", callee_name(callee)),
        InstKind::Const { .. } => "const",
        InstKind::ConstStr { .. } => "const_str",
        InstKind::StringConcat { .. } => "concat",
        InstKind::StringAppend { .. } => "append",
        InstKind::StringEq { .. } => "string_eq",
        InstKind::StringClone { .. } => "string_clone",
        InstKind::StructuralEq { .. } => "eq",
        InstKind::StructuralClone { .. } => "clone",
        InstKind::Ref { .. } => "ref",
        InstKind::Take { .. } => "take",
        InstKind::Assign { .. } => "assign",
        InstKind::AsSlice { .. } => "as_slice",
        InstKind::Index { .. } => "index",
        InstKind::IndexSet { .. } => "index_set",
        InstKind::Fetch { .. } => "fetch",
        InstKind::Commit { .. } => "commit",
        InstKind::FieldGet { .. } => "field_get",
        InstKind::FieldSet { .. } => "field_set",
        InstKind::Cast { .. } => "cast",
        InstKind::LoadFunction { .. } => "load_function",
        InstKind::Eval { .. } => "eval",
        InstKind::Merge { .. } => "merge",
        InstKind::MakeArray { .. } => "make_array",
        InstKind::MakeObject { .. } => "make_object",
        InstKind::MakeTuple { .. } => "make_tuple",
        InstKind::TupleIndex { .. } => "tuple_index",
        InstKind::TestLiteral { .. } => "test_literal",
        InstKind::TestObjectKey { .. } => "test_key",
        InstKind::ArrayIndex { .. } => "array_index",
        InstKind::ObjectGet { .. } => "object_get",
        InstKind::MakeClosure { .. } => "make_closure",
        InstKind::MakeVariant { .. } => "make_variant",
        InstKind::TestVariant { .. } => "test_variant",
        InstKind::UnwrapVariant { .. } => "unwrap_variant",
        InstKind::BlockLabel { .. } => "label",
        InstKind::Jump { .. } => "jump",
        InstKind::JumpIf { .. } | InstKind::Diamond { .. } => "branch",
        InstKind::Switch { .. } => "switch",
        InstKind::For { .. } => "for",
        InstKind::Return { .. } => "return",
        InstKind::Diverge => "diverge",
        InstKind::Undef { .. } => "undef",
        InstKind::Nop => "nop",
        InstKind::Drop { .. } => "drop",
        InstKind::Poison { .. } => "poison",
    };
    name.to_string()
}

fn fmt_label(l: Label) -> String {
    format!("L{}", l.0)
}

fn fmt_str(text: &str) -> String {
    format!("{text:?}")
}

fn fmt_literal(lit: &Literal) -> String {
    match lit {
        Literal::Int(n) => n.to_string(),
        Literal::Float(f) => format!("{f:?}"),
        Literal::Char(c) => format!("{c:?}"),
        sugar @ (Literal::IntOf(_) | Literal::Bytes(_)) => fmt_literal(&sugar.desugared()),
        Literal::String(s) => fmt_str(s),
        Literal::Bool(b) => b.to_string(),
        Literal::Unit => "()".to_string(),
        Literal::List(elems) => {
            let items: Vec<String> = elems.iter().map(fmt_literal).collect();
            format!("[{}]", items.join(", "))
        }
    }
}

enum Spelling {
    Infix(&'static str),
    Call(&'static str),
}

/// A trapping operation is spelled as the source spells it; a wrapping one,
/// which only a pass writes, carries `%` after its operator, as Zig spells
/// its wrapping operators (`+%`, `-%`, `*%`).
fn fmt_binop(op: BinOp) -> Spelling {
    match op {
        BinOp::Add(Overflow::Trap) => Spelling::Infix("+"),
        BinOp::Sub(Overflow::Trap) => Spelling::Infix("-"),
        BinOp::Mul(Overflow::Trap) => Spelling::Infix("*"),
        BinOp::Add(Overflow::Wrap) => Spelling::Infix("+%"),
        BinOp::Sub(Overflow::Wrap) => Spelling::Infix("-%"),
        BinOp::Mul(Overflow::Wrap) => Spelling::Infix("*%"),
        BinOp::Div => Spelling::Infix("/"),
        BinOp::Eq => Spelling::Infix("=="),
        BinOp::Neq => Spelling::Infix("!="),
        BinOp::Lt => Spelling::Infix("<"),
        BinOp::Gt => Spelling::Infix(">"),
        BinOp::Lte => Spelling::Infix("<="),
        BinOp::Gte => Spelling::Infix(">="),
        BinOp::And => Spelling::Infix("&&"),
        BinOp::Or => Spelling::Infix("||"),
        BinOp::Xor => Spelling::Infix("^"),
        BinOp::BitAnd => Spelling::Infix("&"),
        BinOp::BitOr => Spelling::Infix("|"),
        BinOp::Shl(Overflow::Trap) => Spelling::Infix("<<"),
        BinOp::Shr(Overflow::Trap) => Spelling::Infix(">>"),
        BinOp::Shl(Overflow::Wrap) => Spelling::Infix("<<%"),
        BinOp::Shr(Overflow::Wrap) => Spelling::Infix(">>%"),
        BinOp::Mod => Spelling::Infix("%"),
        BinOp::Min => Spelling::Call("min"),
        BinOp::Max => Spelling::Call("max"),
    }
}

fn fmt_unaryop(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg(Overflow::Trap) => "-",
        UnaryOp::Neg(Overflow::Wrap) => "-%",
        UnaryOp::Not => "!",
    }
}

struct PrintCtx<'a> {
    interner: &'a Interner,
    facts: StageFacts<'a>,
    lit_to_tidx: &'a FxHashMap<String, usize>,
    /// FunctionId -> canonical index (order of first appearance across all bodies).
    fn_id_map: FxHashMap<crate::graph::QualifiedRef, usize>,
}

impl PrintCtx<'_> {
    fn tag_name(&self, tag: &Astr) -> String {
        self.interner.resolve(*tag).to_string()
    }

    fn fmt_fn_id(&self, id: crate::graph::QualifiedRef) -> String {
        match self.fn_id_map.get(&id) {
            Some(&idx) => format!("#{idx}"),
            None => {
                let name = self.interner.resolve(id.name);
                format!("#?{name}")
            }
        }
    }
}

/// Collect FunctionIds from a body in order of first appearance.
fn collect_fn_ids_from_body(
    body: &MirBody,
    fn_id_map: &mut FxHashMap<crate::graph::QualifiedRef, usize>,
) {
    for inst in &body.insts {
        let ids: &[crate::graph::QualifiedRef] = match &inst.kind {
            InstKind::LoadFunction { id, .. } => std::slice::from_ref(id),
            InstKind::FunctionCall {
                callee: Callee::Direct(id) | Callee::Extern { id, .. },
                ..
            } => std::slice::from_ref(id),
            InstKind::Spawn {
                callee: Callee::Direct(id) | Callee::Extern { id, .. },
                ..
            } => std::slice::from_ref(id),
            _ => &[],
        };
        for &id in ids {
            let len = fn_id_map.len();
            fn_id_map.entry(id).or_insert(len);
        }
    }
}

/// Collect unique String/List literals from a body and register them into the text table.
fn collect_texts_from_body(
    body: &MirBody,
    lit_to_tidx: &mut FxHashMap<String, usize>,
    text_entries: &mut Vec<String>,
) {
    for inst in &body.insts {
        let key = match &inst.kind {
            InstKind::Const { value, .. }
                if matches!(value, Literal::String(_) | Literal::List(_)) =>
            {
                fmt_literal(value)
            }
            InstKind::ConstStr { text, .. } => fmt_str(text),
            _ => continue,
        };
        if !lit_to_tidx.contains_key(&key) {
            let idx = text_entries.len();
            lit_to_tidx.insert(key.clone(), idx);
            text_entries.push(key);
        }
    }
}

/// A storage and a field path under it, as `Ref`, `Take`, and `Assign` name one.
fn fmt_place(
    body: &MirBody,
    ctx: &PrintCtx<'_>,
    vn: &mut ValNormalizer,
    target: &crate::ir::RefTarget,
    path: &[crate::ir::PathSeg],
) -> String {
    let base = match target {
        crate::ir::RefTarget::Var(slot) => body.debug.label(*slot, ctx.interner),
        crate::ir::RefTarget::Param(slot) => {
            let name = body.debug.label(*slot, ctx.interner);
            if name.starts_with('$') {
                name
            } else {
                format!("${name}")
            }
        }
        crate::ir::RefTarget::Through(r) => format!("(*{})", vn.fmt_val(*r)),
    };
    if path.is_empty() {
        base
    } else {
        let suffix: String = path
            .iter()
            .map(|seg| match seg {
                crate::ir::PathSeg::Field(f) => format!(".{}", ctx.interner.resolve(*f)),
                crate::ir::PathSeg::Index(i) => format!("[{i}]"),
                crate::ir::PathSeg::Payload => ".payload".to_string(),
            })
            .collect();
        format!("{base}{suffix}")
    }
}

/// A body's `For`s as `analysis::loop_deps` computes them, for the
/// comment lines under each.
struct Computed<'a> {
    cfg: CfgBody,
    deps: BodyDeps,
    laws: &'a LawTable,
    costs: Option<&'a CostTable>,
}

fn write_body(
    f: &mut fmt::Formatter<'_>,
    body: &MirBody,
    indent: &str,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    let mut vn = ValNormalizer::new();
    let computed = match ctx.facts {
        StageFacts::Boundaries => None,
        StageFacts::Computed { laws, costs } => {
            let cfg = promote(body.clone());
            Some(Computed {
                deps: BodyDeps::of(&cfg, laws),
                cfg,
                laws,
                costs,
            })
        }
    };
    let costs = computed.as_ref().and_then(|computed| {
        computed
            .costs
            .map(|table| Costs::of(&computed.cfg, computed.laws, table))
    });
    let mut block = crate::cfg::ENTRY_LABEL;

    // A constant is shown at its use sites only while its ValueId names that
    // one definition; after register allocation an id is reused, and a use
    // of a reused id prints as the id.
    let mut def_count: FxHashMap<ValueId, usize> = FxHashMap::default();
    for inst in &body.insts {
        for dst in crate::analysis::inst_info::defs(&inst.kind) {
            *def_count.entry(dst).or_default() += 1;
        }
    }
    let single_consts = body.insts.iter().filter_map(|inst| match &inst.kind {
        InstKind::Const { dst, value } if def_count[dst] == 1 => Some((*dst, value)),
        _ => None,
    });

    // Small constants (Int, Float, Bool, Byte) -> inline at use sites.
    let consts: FxHashMap<ValueId, &Literal> = single_consts
        .clone()
        .filter(|(_, value)| !matches!(value, Literal::String(_) | Literal::List(_)))
        .collect();

    // String/List constants -> T-indexed references.
    let texts: FxHashMap<ValueId, usize> = single_consts
        .filter(|(_, value)| matches!(value, Literal::String(_) | Literal::List(_)))
        .filter_map(|(dst, value)| {
            ctx.lit_to_tidx
                .get(&fmt_literal(value))
                .map(|&tidx| (dst, tidx))
        })
        .collect();

    for (i, inst) in body.insts.iter().enumerate() {
        if let InstKind::Const { dst, .. } = &inst.kind
            && (consts.contains_key(dst) || texts.contains_key(dst))
        {
            continue;
        }

        let is_label = matches!(&inst.kind, InstKind::BlockLabel { .. });
        // Fixed-width index column, then content indent for non-labels.
        if is_label {
            write!(f, "{indent}{i:>4} | ")?;
        } else {
            write!(f, "{indent}{i:>4} |   ")?;
        }

        match &inst.kind {
            InstKind::Const { dst, value } => {
                let shown = match ctx.lit_to_tidx.get(&fmt_literal(value)) {
                    Some(tidx) if matches!(value, Literal::String(_) | Literal::List(_)) => {
                        format!("T{tidx}")
                    }
                    _ => fmt_literal(value),
                };
                writeln!(f, "{} = const {}", vn.fmt_val(*dst), shown)?
            }
            InstKind::ConstStr { dst, text } => {
                let key = fmt_str(text);
                let shown = match ctx.lit_to_tidx.get(&key) {
                    Some(tidx) => format!("T{tidx}"),
                    None => key,
                };
                writeln!(f, "{} = const_str {}", vn.fmt_val(*dst), shown)?
            }
            // Projection
            InstKind::Ref {
                dst,
                target,
                path,
                mutability,
            } => writeln!(
                f,
                "{} = ref {}{}",
                vn.fmt_val(*dst),
                mutability.prefix(),
                fmt_place(body, ctx, &mut vn, target, path)
            )?,
            InstKind::Take {
                dst,
                target,
                path,
                taken_out,
            } => writeln!(
                f,
                "{} = {} {}",
                vn.fmt_val(*dst),
                if *taken_out { "take-out" } else { "take" },
                fmt_place(body, ctx, &mut vn, target, path)
            )?,
            InstKind::Assign {
                target,
                path,
                value,
                restores,
            } => writeln!(
                f,
                "{} {} = {}",
                if *restores { "restore" } else { "assign" },
                fmt_place(body, ctx, &mut vn, target, path),
                vn.fmt_use(*value, &consts, &texts)
            )?,
            InstKind::Fetch { dst, context } => writeln!(
                f,
                "{} = fetch @{}",
                vn.fmt_val(*dst),
                ctx.interner.resolve(context.name)
            )?,
            InstKind::Commit {
                context,
                value,
                wrote,
            } => writeln!(
                f,
                "commit @{} = {}{}",
                ctx.interner.resolve(context.name),
                vn.fmt_use(*value, &consts, &texts),
                if *wrote { "" } else { " (not written)" }
            )?,

            // Scalar field access
            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let mut path = ctx.interner.resolve(*field).to_string();
                for r in rest {
                    path.push('.');
                    path.push_str(ctx.interner.resolve(*r));
                }
                writeln!(
                    f,
                    "{} = {}.{}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*object, &consts, &texts),
                    path
                )?
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                let mut path = ctx.interner.resolve(*field).to_string();
                for r in rest {
                    path.push('.');
                    path.push_str(ctx.interner.resolve(*r));
                }
                writeln!(
                    f,
                    "{} = field_set {}.{} = {}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*object, &consts, &texts),
                    path,
                    vn.fmt_use(*value, &consts, &texts)
                )?
            }

            // Arithmetic / logic
            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => {
                let (dst, left, right) = (
                    vn.fmt_val(*dst),
                    vn.fmt_use(*left, &consts, &texts),
                    vn.fmt_use(*right, &consts, &texts),
                );
                match fmt_binop(*op) {
                    Spelling::Infix(op) => writeln!(f, "{dst} = {left} {op} {right}")?,
                    Spelling::Call(name) => writeln!(f, "{dst} = {name}({left}, {right})")?,
                }
            }
            InstKind::UnaryOp { dst, op, operand } => writeln!(
                f,
                "{} = {}{}",
                vn.fmt_val(*dst),
                fmt_unaryop(*op),
                vn.fmt_use(*operand, &consts, &texts)
            )?,
            InstKind::Check { op, left, right } => {
                let Spelling::Infix(spelled) = fmt_binop(op.trapping_op()) else {
                    panic!("{op:?}'s operation is spelled infix")
                };
                writeln!(
                    f,
                    "check {} {spelled} {}",
                    vn.fmt_use(*left, &consts, &texts),
                    vn.fmt_use(*right, &consts, &texts)
                )?
            }
            InstKind::CheckSteps { from, step, count } => writeln!(
                f,
                "check {} + {} * {}",
                vn.fmt_use(*from, &consts, &texts),
                vn.fmt_use(*count, &consts, &texts),
                vn.fmt_use(*step, &consts, &texts)
            )?,
            InstKind::Cast { dst, src, to } => writeln!(
                f,
                "{} = {} as {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                to.name()
            )?,

            // Functions
            InstKind::LoadFunction { dst, id } => writeln!(
                f,
                "{} = load_function {}",
                vn.fmt_val(*dst),
                ctx.fmt_fn_id(*id),
            )?,
            InstKind::FunctionCall {
                dst,
                callee,
                args,
                order,
                ..
            } => {
                let callee_str = match callee {
                    Callee::Direct(id) | Callee::Extern { id, .. } => ctx.fmt_fn_id(*id),
                    Callee::Indirect(val) => vn.fmt_use(*val, &consts, &texts),
                };
                write!(
                    f,
                    "{} = call {}({})",
                    vn.fmt_val(*dst),
                    callee_str,
                    vn.fmt_uses(args, &consts, &texts)
                )?;
                if let Some(edge) = order {
                    write!(
                        f,
                        " [{} -> {}]",
                        vn.fmt_use(edge.before, &consts, &texts),
                        vn.fmt_val(edge.after)
                    )?;
                }
                writeln!(f)?
            }
            InstKind::Merge { dst, orders } => writeln!(
                f,
                "{} = merge {}",
                vn.fmt_val(*dst),
                vn.fmt_uses(orders, &consts, &texts)
            )?,

            // Spawn / Eval
            InstKind::Spawn {
                dst,
                callee,
                args,
                order,
                ..
            } => {
                let callee_str = match callee {
                    Callee::Direct(id) | Callee::Extern { id, .. } => ctx.fmt_fn_id(*id),
                    Callee::Indirect(val) => vn.fmt_use(*val, &consts, &texts),
                };
                write!(
                    f,
                    "{} = spawn {}({})",
                    vn.fmt_val(*dst),
                    callee_str,
                    vn.fmt_uses(args, &consts, &texts)
                )?;
                if let Some(o) = order {
                    write!(f, " [{} ->]", vn.fmt_use(*o, &consts, &texts))?;
                }
                writeln!(f)?
            }
            InstKind::Eval { dst, src, order } => {
                write!(
                    f,
                    "{} = eval {}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*src, &consts, &texts)
                )?;
                if let Some(o) = order {
                    write!(f, " [-> {}]", vn.fmt_val(*o))?;
                }
                writeln!(f)?
            }

            // Composite constructors
            InstKind::StringClone { dst, src } => writeln!(
                f,
                "{} = string_clone {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts)
            )?,
            InstKind::StringEq { dst, a, b } => writeln!(
                f,
                "{} = string_eq {} {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*a, &consts, &texts),
                vn.fmt_use(*b, &consts, &texts)
            )?,
            InstKind::StructuralEq { dst, a, b, leaves } => writeln!(
                f,
                "{} = structural_eq {} {} leaves={}",
                vn.fmt_val(*dst),
                vn.fmt_use(*a, &consts, &texts),
                vn.fmt_use(*b, &consts, &texts),
                leaves.len()
            )?,
            InstKind::StructuralClone { dst, src, leaves } => writeln!(
                f,
                "{} = structural_clone {} leaves={}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                leaves.len()
            )?,
            InstKind::StringConcat { dst, parts } => writeln!(
                f,
                "{} = string_concat [{}]",
                vn.fmt_val(*dst),
                vn.fmt_uses(parts, &consts, &texts)
            )?,
            InstKind::StringAppend { target, part } => writeln!(
                f,
                "append {} {}",
                vn.fmt_use(*target, &consts, &texts),
                vn.fmt_use(*part, &consts, &texts)
            )?,
            InstKind::MakeArray { dst, elements } => writeln!(
                f,
                "{} = list [{}]",
                vn.fmt_val(*dst),
                vn.fmt_uses(elements, &consts, &texts)
            )?,
            InstKind::MakeObject { dst, fields } => {
                let fields_str: String = fields
                    .iter()
                    .map(|(k, r)| {
                        format!(
                            "{}: {}",
                            ctx.interner.resolve(*k),
                            vn.fmt_use(*r, &consts, &texts)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "{} = object {{{fields_str}}}", vn.fmt_val(*dst))?
            }
            InstKind::MakeTuple { dst, elements } => writeln!(
                f,
                "{} = tuple ({})",
                vn.fmt_val(*dst),
                vn.fmt_uses(elements, &consts, &texts)
            )?,
            InstKind::TupleIndex { dst, tuple, index } => writeln!(
                f,
                "{} = {}.{index}",
                vn.fmt_val(*dst),
                vn.fmt_use(*tuple, &consts, &texts)
            )?,

            // Pattern matching
            InstKind::TestLiteral { dst, src, value } => writeln!(
                f,
                "{} = test {} == {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                fmt_literal(value)
            )?,
            InstKind::TestObjectKey { dst, src, key } => writeln!(
                f,
                "{} = test has_key({}, \"{}\")",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                ctx.interner.resolve(*key),
            )?,
            InstKind::AsSlice {
                dst,
                container,
                mutability,
                ..
            } => writeln!(
                f,
                "{} = as_slice {}{}",
                vn.fmt_val(*dst),
                mutability.prefix(),
                vn.fmt_use(*container, &consts, &texts)
            )?,
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
                bound,
            } => writeln!(
                f,
                "{} = {}{}[{}]{}",
                vn.fmt_val(*dst),
                match mode {
                    IndexMode::Copy => "",
                    IndexMode::Ref => "&",
                },
                vn.fmt_use(*slice, &consts, &texts),
                vn.fmt_use(*index, &consts, &texts),
                proven_suffix(*bound)
            )?,
            InstKind::IndexSet {
                slice,
                index,
                value,
                bound,
            } => writeln!(
                f,
                "{}[{}]{} = {}",
                vn.fmt_use(*slice, &consts, &texts),
                vn.fmt_use(*index, &consts, &texts),
                proven_suffix(*bound),
                vn.fmt_use(*value, &consts, &texts)
            )?,
            InstKind::ArrayIndex {
                dst,
                array: list,
                index,
            } => writeln!(
                f,
                "{} = {}[{index}]",
                vn.fmt_val(*dst),
                vn.fmt_use(*list, &consts, &texts)
            )?,
            InstKind::ObjectGet { dst, object, key } => writeln!(
                f,
                "{} = {}.{}",
                vn.fmt_val(*dst),
                vn.fmt_use(*object, &consts, &texts),
                ctx.interner.resolve(*key),
            )?,

            // Variant
            InstKind::MakeVariant { dst, tag, payload } => {
                let name = ctx.tag_name(tag);
                match payload {
                    Some(p) => writeln!(
                        f,
                        "{} = variant {}({})",
                        vn.fmt_val(*dst),
                        name,
                        vn.fmt_use(*p, &consts, &texts)
                    )?,
                    None => writeln!(f, "{} = variant {}", vn.fmt_val(*dst), name)?,
                }
            }
            InstKind::TestVariant { dst, src, tag } => {
                let name = ctx.tag_name(tag);
                writeln!(
                    f,
                    "{} = test {} is {}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*src, &consts, &texts),
                    name,
                )?
            }
            InstKind::UnwrapVariant { dst, src } => writeln!(
                f,
                "{} = unwrap {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts)
            )?,

            // Closures
            InstKind::MakeClosure {
                dst,
                body,
                captures,
            } => writeln!(
                f,
                "{} = closure {} [{}]",
                vn.fmt_val(*dst),
                fmt_label(*body),
                vn.fmt_uses(captures, &consts, &texts)
            )?,

            // Iteration

            // Control flow
            InstKind::BlockLabel { label, params } => {
                block = *label;
                if params.is_empty() {
                    writeln!(f, "{}:", fmt_label(*label))?
                } else {
                    let params_str = params
                        .iter()
                        .map(|v| {
                            let ty = body
                                .val_types
                                .get(v)
                                .map(|t| format!("{}", t.display(ctx.interner)))
                                .unwrap_or_else(|| "?".into());
                            format!("{}: {ty}", vn.fmt_val(*v))
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(f, "{}({params_str}):", fmt_label(*label))?
                }
            }
            // `for slice(r3) -> L1 else L2 stages [L1, L3]` (RFC-0057,
            // RFC-0089 rule 1), and with computed facts a comment line per
            // stage below it. The element and the counter are the body
            // block's parameters, printed where that block's label is. An exit edge
            // that defines the trip count prints it as `trip` where the exit
            // block's first parameter takes it: `else L2(trip, r5)`
            // (RFC-0057 rule 9).
            InstKind::For {
                source,
                stages,
                exit,
                exit_trip,
                exit_args,
            } => {
                let (source, exit, exit_trip) = (*source, *exit, *exit_trip);
                let stages_body = stages.body();
                let over = match source {
                    ForSource::Slice(slice) => {
                        format!("slice({})", vn.fmt_use(slice, &consts, &texts))
                    }
                    ForSource::SliceMut(slice) => {
                        format!("slice_mut({})", vn.fmt_use(slice, &consts, &texts))
                    }
                    ForSource::Array(array) => {
                        format!("array({})", vn.fmt_use(array, &consts, &texts))
                    }
                    ForSource::Range { at, hi } => format!(
                        "range({}..{})",
                        vn.fmt_use(at, &consts, &texts),
                        vn.fmt_use(hi, &consts, &texts)
                    ),
                };
                let carried =
                    (!exit_args.is_empty()).then(|| vn.fmt_uses(exit_args, &consts, &texts));
                let left = match (exit_trip, carried) {
                    (ExitTrip::Absent, None) => fmt_label(exit),
                    (ExitTrip::Absent, Some(carried)) => {
                        format!("{}({carried})", fmt_label(exit))
                    }
                    (ExitTrip::Defined, None) => format!("{}(trip)", fmt_label(exit)),
                    (ExitTrip::Defined, Some(carried)) => {
                        format!("{}(trip, {carried})", fmt_label(exit))
                    }
                };
                let entries: Vec<String> = stages.entries().map(fmt_label).collect();
                writeln!(
                    f,
                    "for {over} -> {} else {left} stages [{}]",
                    fmt_label(stages_body),
                    entries.join(", ")
                )?;
                if let Some(computed) = &computed {
                    let found = computed
                        .deps
                        .loops
                        .iter()
                        .find(|found| computed.cfg.blocks[found.header.0].label == block)
                        .expect("promoting a body keeps each `For` at the end of its block");
                    let lines = match &found.deps {
                        Ok(deps) => {
                            let mut lines =
                                fmt_loop_facts(deps, &computed.cfg, computed.laws, ctx, &mut vn);
                            if let Some(costs) = &costs {
                                let trip = costs.trip(found.header).and_then(|trip| match trip {
                                    Trip::Known(term) => {
                                        Some(fmt_term(term, &mut vn, &consts, &texts))
                                    }
                                    Trip::Unknown => None,
                                });
                                lines.push(fmt_cost(costs.of_loop(deps), trip));
                            }
                            lines
                        }
                        Err(fault) => vec![format!("stages refused: {}", fault.shown())],
                    };
                    for line in lines {
                        writeln!(f, "{indent}     |     // {line}")?;
                    }
                }
            }
            // `switch r5 { A -> L1, B -> L2, _ -> L3 }`, and a literal
            // dispatch's keys as written: `switch r5 { 1 -> L1, _ -> L2 }`
            // (RFC-0051).
            InstKind::Switch { tag, arms, default } => {
                let mut edge = |label: &Label, args: &[ValueId]| {
                    if args.is_empty() {
                        fmt_label(*label)
                    } else {
                        format!(
                            "{}({})",
                            fmt_label(*label),
                            vn.fmt_uses(args, &consts, &texts)
                        )
                    }
                };
                let mut parts: Vec<String> = arms
                    .iter()
                    .map(|(key, label, args)| {
                        format!("{} -> {}", key.shown(ctx.interner), edge(label, args))
                    })
                    .collect();
                if let Some((label, args)) = default {
                    parts.push(format!("_ -> {}", edge(label, args)));
                }
                drop(edge);
                writeln!(
                    f,
                    "switch {} {{ {} }}",
                    vn.fmt_use(*tag, &consts, &texts),
                    parts.join(", ")
                )?
            }
            InstKind::Jump { label, args } => {
                if args.is_empty() {
                    writeln!(f, "jump {}", fmt_label(*label))?
                } else {
                    writeln!(
                        f,
                        "jump {}({})",
                        fmt_label(*label),
                        vn.fmt_uses(args, &consts, &texts)
                    )?
                }
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let then_str = if then_args.is_empty() {
                    fmt_label(*then_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*then_label),
                        vn.fmt_uses(then_args, &consts, &texts)
                    )
                };
                let else_str = if else_args.is_empty() {
                    fmt_label(*else_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*else_label),
                        vn.fmt_uses(else_args, &consts, &texts)
                    )
                };
                writeln!(
                    f,
                    "jump_if {} then {} else {}",
                    vn.fmt_use(*cond, &consts, &texts),
                    then_str,
                    else_str
                )?
            }
            InstKind::Diamond {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
                join,
            } => {
                let cond_str = vn.fmt_use(*cond, &consts, &texts);
                let then_str = if then_args.is_empty() {
                    fmt_label(*then_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*then_label),
                        vn.fmt_uses(then_args, &consts, &texts)
                    )
                };
                let else_str = if else_args.is_empty() {
                    fmt_label(*else_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*else_label),
                        vn.fmt_uses(else_args, &consts, &texts)
                    )
                };
                writeln!(
                    f,
                    "if {} -> {} else {} join {}",
                    cond_str,
                    then_str,
                    else_str,
                    fmt_label(*join)
                )?
            }
            InstKind::Return { value, order } => {
                write!(f, "return {}", vn.fmt_use(*value, &consts, &texts))?;
                if let Some(o) = order {
                    write!(f, " [{}]", vn.fmt_use(*o, &consts, &texts))?;
                }
                writeln!(f)?
            }
            InstKind::Diverge => writeln!(f, "diverge")?,
            InstKind::Nop => writeln!(f, "nop")?,
            InstKind::Drop { src } => writeln!(f, "drop {}", vn.fmt_use(*src, &consts, &texts))?,
            InstKind::Undef { dst } => writeln!(f, "{} = undef", vn.fmt_val(*dst))?,
            InstKind::Poison { dst } => writeln!(f, "{} = poison", vn.fmt_val(*dst))?,
        }
    }

    write_order_tree(f, body, indent, ctx, &mut vn)?;

    // Print value types with origin names.
    if !body.val_types.is_empty() {
        writeln!(f)?;
        let mut entries: Vec<_> = body.val_types.iter().collect();
        entries.sort_by_key(|(v, _)| vn.get(**v));
        for (val, ty) in entries {
            let origin = body.debug.label(*val, ctx.interner);
            writeln!(
                f,
                "{indent}  ; {} ({origin}) : {}",
                vn.fmt_val(*val),
                ty.display(ctx.interner),
            )?;
        }
    }

    Ok(())
}

/// Display wrapper for MirModule that requires an interner.
pub struct MirModuleDisplay<'a> {
    module: &'a MirModule,
    interner: &'a Interner,
    facts: StageFacts<'a>,
}

impl fmt::Display for MirModuleDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let module = self.module;

        // Collect unique String/List literals across all bodies -> text table.
        let mut lit_to_tidx: FxHashMap<String, usize> = FxHashMap::default();
        let mut text_entries: Vec<String> = Vec::new();

        collect_texts_from_body(&module.main, &mut lit_to_tidx, &mut text_entries);
        let mut labels: Vec<_> = module.closures.keys().collect();
        labels.sort_by_key(|l| l.0);
        for label in &labels {
            collect_texts_from_body(&module.closures[label], &mut lit_to_tidx, &mut text_entries);
        }

        // Collect FunctionIds across all bodies for canonical numbering.
        let mut fn_id_map: FxHashMap<crate::graph::QualifiedRef, usize> = FxHashMap::default();
        collect_fn_ids_from_body(&module.main, &mut fn_id_map);
        for label in &labels {
            collect_fn_ids_from_body(&module.closures[label], &mut fn_id_map);
        }

        if !text_entries.is_empty() {
            writeln!(f, "=== literals ===")?;
            for (idx, lit) in text_entries.iter().enumerate() {
                writeln!(f, "  T{idx} = {lit}")?;
            }
            writeln!(f)?;
        }

        let ctx = PrintCtx {
            interner: self.interner,
            facts: self.facts,
            lit_to_tidx: &lit_to_tidx,
            fn_id_map,
        };

        writeln!(f, "=== main ===")?;
        write_body(f, &module.main, "  ", &ctx)?;

        for label in &labels {
            let closure = &module.closures[label];
            write_closure(f, **label, closure, &ctx)?;
        }

        Ok(())
    }
}

impl MirModule {
    /// Create a display wrapper that resolves Astr values via the interner.
    pub fn display<'a>(&'a self, interner: &'a Interner) -> MirModuleDisplay<'a> {
        MirModuleDisplay {
            module: self,
            interner,
            facts: StageFacts::Boundaries,
        }
    }

    /// As [`Self::display`], with what `analysis::loop_deps` computes of
    /// each `For`'s stages under its terminator.
    pub fn display_with_facts<'a>(
        &'a self,
        interner: &'a Interner,
        laws: &'a LawTable,
    ) -> MirModuleDisplay<'a> {
        MirModuleDisplay {
            module: self,
            interner,
            facts: StageFacts::Computed { laws, costs: None },
        }
    }

    /// As [`Self::display_with_facts`], with what `analysis::cost` computes
    /// of each `For` against a backend's `table` (RFC-0066 rule 8).
    pub fn display_with_costs<'a>(
        &'a self,
        interner: &'a Interner,
        laws: &'a LawTable,
        table: &'a CostTable,
    ) -> MirModuleDisplay<'a> {
        MirModuleDisplay {
            module: self,
            interner,
            facts: StageFacts::Computed {
                laws,
                costs: Some(table),
            },
        }
    }
}

#[derive(Clone)]
struct OrderNode {
    /// Named by instruction index: registers are reused after coloring.
    label: String,
    waits_for: Vec<OrderNode>,
}

fn write_order_tree(
    f: &mut fmt::Formatter<'_>,
    body: &MirBody,
    indent: &str,
    ctx: &PrintCtx<'_>,
    vn: &mut ValNormalizer,
) -> fmt::Result {
    let mut produced: FxHashMap<ValueId, OrderNode> = FxHashMap::default();
    let entry = |v: ValueId, vn: &mut ValNormalizer| OrderNode {
        label: format!("{}(entry)", vn.fmt_val(v)),
        waits_for: Vec::new(),
    };
    let mut handles: FxHashMap<ValueId, OrderNode> = FxHashMap::default();
    let mut returned: Option<OrderNode> = None;
    let mut any = false;
    for (i, inst) in body.insts.iter().enumerate() {
        let callee_name = |c: &Callee, vn: &mut ValNormalizer| match c {
            Callee::Direct(id) | Callee::Extern { id, .. } => ctx.fmt_fn_id(*id),
            Callee::Indirect(v) => vn.fmt_val(*v),
        };
        match &inst.kind {
            InstKind::FunctionCall {
                callee,
                order: Some(edge),
                ..
            } => {
                any = true;
                let before = produced
                    .get(&edge.before)
                    .cloned()
                    .unwrap_or_else(|| entry(edge.before, vn));
                produced.insert(
                    edge.after,
                    OrderNode {
                        label: format!("call@{i} {}", callee_name(callee, vn)),
                        waits_for: vec![before],
                    },
                );
            }
            InstKind::Spawn {
                dst,
                callee,
                order: Some(o),
                ..
            } => {
                any = true;
                let before = produced.get(o).cloned().unwrap_or_else(|| entry(*o, vn));
                handles.insert(
                    *dst,
                    OrderNode {
                        label: format!("spawn@{i} {}", callee_name(callee, vn)),
                        waits_for: vec![before],
                    },
                );
            }
            InstKind::Eval {
                src,
                order: Some(o),
                ..
            } => {
                any = true;
                let spawned = handles.get(src).cloned();
                produced.insert(
                    *o,
                    OrderNode {
                        label: format!("eval@{i}"),
                        waits_for: spawned.into_iter().collect(),
                    },
                );
            }
            InstKind::Merge { dst, orders } => {
                any = true;
                let waits_for = orders
                    .iter()
                    .map(|o| produced.get(o).cloned().unwrap_or_else(|| entry(*o, vn)))
                    .collect();
                produced.insert(
                    *dst,
                    OrderNode {
                        label: format!("merge@{i}"),
                        waits_for,
                    },
                );
            }
            InstKind::Return { order: Some(o), .. } => {
                returned = Some(produced.get(o).cloned().unwrap_or_else(|| entry(*o, vn)));
            }
            _ => {}
        }
    }
    if !any {
        return Ok(());
    }
    let Some(root) = returned else {
        return Ok(());
    };
    writeln!(f)?;
    writeln!(
        f,
        "{indent}  ; orders: what the body's yielded Order waits for"
    )?;
    write_order_node(f, &root, indent, 0)
}

fn write_order_node(
    f: &mut fmt::Formatter<'_>,
    node: &OrderNode,
    indent: &str,
    depth: usize,
) -> fmt::Result {
    let mut line = node.label.clone();
    let mut current = node;
    while current.waits_for.len() == 1 {
        current = &current.waits_for[0];
        line.push_str(" <- ");
        line.push_str(&current.label);
    }
    writeln!(f, "{indent}  ;   {}{line}", "  ".repeat(depth))?;
    for child in &current.waits_for {
        write_order_node(f, child, indent, depth + 1)?;
    }
    Ok(())
}

fn write_closure(
    f: &mut fmt::Formatter<'_>,
    label: Label,
    body: &MirBody,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    writeln!(f)?;
    write!(f, "=== closure {} (", fmt_label(label))?;
    for (i, (_, reg)) in body.params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{reg:?}")?;
    }
    write!(f, ")")?;
    if !body.captures.is_empty() {
        let cap_regs: Vec<_> = body.captures.iter().map(|(_, v)| v).collect();
        write!(f, " [captures: {:?}]", cap_regs)?;
    }
    writeln!(f, " ===")?;
    write_body(f, body, "  ", ctx)?;
    Ok(())
}

/// Display wrapper for MirBody that requires an interner.
pub struct MirBodyDisplay<'a> {
    body: &'a MirBody,
    interner: &'a Interner,
}

impl fmt::Display for MirBodyDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let empty_lits = FxHashMap::default();
        let mut fn_id_map = FxHashMap::default();
        collect_fn_ids_from_body(self.body, &mut fn_id_map);
        let ctx = PrintCtx {
            interner: self.interner,
            facts: StageFacts::Boundaries,
            lit_to_tidx: &empty_lits,
            fn_id_map,
        };
        write_body(f, self.body, "", &ctx)
    }
}

impl MirBody {
    /// Create a display wrapper that resolves Astr values via the interner.
    pub fn display<'a>(&'a self, interner: &'a Interner) -> MirBodyDisplay<'a> {
        MirBodyDisplay {
            body: self,
            interner,
        }
    }
}

/// Dump a MirModule to a String, resolving Astr values via the interner.
pub fn dump(interner: &Interner, module: &MirModule) -> String {
    format!("{}", module.display(interner))
}

/// Alias for `dump`. Kept for backward compatibility.
pub fn dump_with(interner: &Interner, module: &MirModule) -> String {
    dump(interner, module)
}

/// `dump`, with each `For`'s computed stage facts.
pub fn dump_with_facts(interner: &Interner, module: &MirModule, laws: &LawTable) -> String {
    format!("{}", module.display_with_facts(interner, laws))
}

/// `dump_with_facts`, with each `For`'s cost against `table`.
pub fn dump_with_costs(
    interner: &Interner,
    module: &MirModule,
    laws: &LawTable,
    table: &CostTable,
) -> String {
    format!("{}", module.display_with_costs(interner, laws, table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{ObjectTy, Ty};
    use acvus_utils::Interner;

    fn compile_and_dump_ctx(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> String {
        let ctx: Vec<(&str, Ty)> = context
            .iter()
            .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
            .collect();
        crate::test::compile_and_dump(interner, source, &ctx)
    }

    #[test]
    fn print_text_only() {
        let interner = Interner::new();
        let out = crate::test::compile_and_dump(&interner, "hello world", &[]);
        assert!(out.contains("=== literals ==="));
        assert!(out.contains("\"hello world\""));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_string_emit() {
        let interner = Interner::new();
        let out = crate::test::compile_and_dump(&interner, r#"{{ "hello" }}"#, &[]);
        assert!(out.contains("\"hello\""));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_if_block() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let out = compile_and_dump_ctx("% if @n == 1\nmatched\n% end\n", &context, &interner);
        assert!(!out.contains("iter_init"), "{out}");
        assert!(!out.contains("iter_next"), "{out}");
        assert!(out.contains(" else L1 join L1"), "{out}");
    }

    #[test]
    fn print_object_field() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(
            interner.intern("user"),
            Ty::Object(ObjectTy::written(FxHashMap::from_iter([
                (interner.intern("n"), Ty::I64),
                (interner.intern("age"), Ty::I64),
            ]))),
        )]);
        let out = compile_and_dump_ctx("% let x = @user.age", &context, &interner);
        assert!(out.contains(".age"), "{out}");
    }

    #[test]
    fn extern_param_write_rejected() {
        let interner = Interner::new();
        let result = crate::test::compile_template(&interner, "% $count = 42", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn print_text_dedup() {
        let interner = Interner::new();
        let out = compile_and_dump_ctx(
            r#"{{ "hello" }}{{ "hello" }}"#,
            &FxHashMap::default(),
            &interner,
        );
        // Same literal "hello" should appear only once in literals section.
        let hello_count = out.matches("\"hello\"").count();
        // Once in literals, possibly referenced in instructions.
        assert!(hello_count >= 1);
        assert!(out.contains("return"));
    }
}
