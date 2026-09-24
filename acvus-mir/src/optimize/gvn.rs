//! Dominator-based global value numbering over pure operations, with
//! integer identities simplified before the lookup (RFC-0083).
//!
//! A replaced operation is left in place with no reader, for `dce`.

use acvus_ast::Literal;
use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::{BinOp, InstKind, ValueId};
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::{CastTy, IntTy, Ty};

pub fn run(cfg: &mut CfgBody) {
    let replacements = Numbering::of(cfg).walk(cfg);
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, &replacements);
        }
        apply_subst_terminator(&mut block.terminator, &replacements);
    }
}

fn ty_of(cfg: &CfgBody, value: ValueId) -> &Ty {
    cfg.val_types
        .get(&value)
        .unwrap_or_else(|| panic!("lowering gives {value:?} a type"))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Word {
    Int { width: IntTy, value: i128 },
    Float { bits: u64 },
    Bool(bool),
    Char(char),
}

impl Word {
    fn of_const(ty: &Ty, literal: &Literal) -> Option<Word> {
        match (ty, literal.desugared()) {
            (Ty::Int(width), Literal::Int(value)) => Some(Word::Int {
                width: *width,
                value,
            }),
            (Ty::Float, Literal::Float(value)) => Some(Word::Float {
                bits: value.to_bits(),
            }),
            (Ty::Bool, Literal::Bool(value)) => Some(Word::Bool(value)),
            (Ty::Char, Literal::Char(value)) => Some(Word::Char(value)),
            _ => None,
        }
    }

    fn int_value(self) -> Option<i128> {
        match self {
            Word::Int { value, .. } => Some(value),
            Word::Float { .. } | Word::Bool(_) | Word::Char(_) => None,
        }
    }
}

fn is_scalar(ty: &Ty) -> bool {
    matches!(ty, Ty::Int(_) | Ty::Float | Ty::Bool | Ty::Char)
}

/// A float `+` and `*` do not commute here: given two NaN operands the
/// machine returns the payload of one of them, chosen by operand order.
fn commutes(op: BinOp, operand: &Ty) -> bool {
    match operand {
        Ty::Int(_) => matches!(
            op,
            BinOp::Add(_)
                | BinOp::Mul(_)
                | BinOp::Eq
                | BinOp::Neq
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::Xor
                | BinOp::Min
                | BinOp::Max
        ),
        Ty::Bool => matches!(op, BinOp::Eq | BinOp::Neq | BinOp::Xor),
        Ty::Float | Ty::Char => matches!(op, BinOp::Eq | BinOp::Neq),
        _ => false,
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Part {
    Field { field: Astr, rest: Vec<Astr> },
    Key(Astr),
    TupleElement(usize),
    ArrayElement(usize),
}

/// An operand as a use reads it, and the number its operation is keyed by.
/// The two differ for a constant equal to a dominating one.
#[derive(Clone, Copy)]
struct Operand {
    value: ValueId,
    number: ValueId,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Operands {
    left: ValueId,
    right: ValueId,
}

impl Operands {
    fn ordered(self) -> Operands {
        match self.right < self.left {
            true => Operands {
                left: self.right,
                right: self.left,
            },
            false => self,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Expr {
    Const(Word),
    /// Keyed by the whole operation, its `Overflow` included: a trapping
    /// and a wrapping `+` over the same operands are two entries, so a
    /// replacement never changes which of the two a use reads.
    Binary {
        op: BinOp,
        operands: Operands,
    },
    Cast {
        to: CastTy,
        src: ValueId,
    },
    Part {
        container: ValueId,
        part: Part,
    },
}

enum Found {
    Const(Word),
    Simplified(ValueId),
    Expr(Expr),
}

struct Numbered {
    dst: ValueId,
    found: Found,
}

struct Replacement {
    replaced: ValueId,
    by: ValueId,
}

struct PartRead {
    dst: ValueId,
    container: ValueId,
    part: Part,
}

enum Visit {
    Enter(BlockIdx),
    LeaveKeeping(usize),
}

struct Numbering {
    dom_children: Vec<Vec<BlockIdx>>,
    /// A storage's name holds whatever was last written through it, so it
    /// is not a value: neither an operation reading one nor a constant
    /// written to one is numbered, and no value is replaced by one.
    storages: FxHashSet<ValueId>,
    words: FxHashMap<ValueId, Word>,
    table: FxHashMap<Expr, ValueId>,
    entered: Vec<Expr>,
    replacements: FxHashMap<ValueId, ValueId>,
    /// A constant equal to one a dominating instruction wrote, keyed to
    /// that one. Operations over the two share a key; the constant itself
    /// stays where it is written, since writing it costs nothing and a
    /// shared one would be copied into each header parameter it fills.
    constant_numbers: FxHashMap<ValueId, ValueId>,
}

impl Numbering {
    fn of(cfg: &CfgBody) -> Self {
        let domtree = DomTree::build(cfg);
        let mut dom_children = vec![Vec::new(); cfg.blocks.len()];
        for block in (0..cfg.blocks.len()).map(BlockIdx) {
            if let Some(parent) = domtree.idom(block) {
                dom_children[parent.0].push(block);
            }
        }
        let mut storages = FxHashSet::default();
        let mut words = FxHashMap::default();
        for inst in cfg.blocks.iter().flat_map(|block| &block.insts) {
            match &inst.kind {
                InstKind::Ref { target, .. }
                | InstKind::Take { target, .. }
                | InstKind::Assign { target, .. } => {
                    storages.extend(inst_info::storage(target));
                }
                InstKind::Const { dst, value } => {
                    if let Some(word) = Word::of_const(ty_of(cfg, *dst), value) {
                        words.insert(*dst, word);
                    }
                }
                _ => {}
            }
        }
        Self {
            dom_children,
            storages,
            words,
            table: FxHashMap::default(),
            entered: Vec::new(),
            replacements: FxHashMap::default(),
            constant_numbers: FxHashMap::default(),
        }
    }

    fn walk(mut self, cfg: &CfgBody) -> FxHashMap<ValueId, ValueId> {
        let mut stack: Vec<Visit> = match cfg.blocks.is_empty() {
            true => Vec::new(),
            false => vec![Visit::Enter(BlockIdx(0))],
        };
        while let Some(visit) = stack.pop() {
            match visit {
                Visit::Enter(block) => {
                    stack.push(Visit::LeaveKeeping(self.entered.len()));
                    let children = self.dom_children[block.0].iter().rev();
                    stack.extend(children.map(|child| Visit::Enter(*child)));
                    for inst in &cfg.blocks[block.0].insts {
                        self.number(cfg, &inst.kind);
                    }
                }
                Visit::LeaveKeeping(kept) => {
                    for expr in self.entered.drain(kept..) {
                        self.table
                            .remove(&expr)
                            .expect("an entered expression stays in the table until its block is left");
                    }
                }
            }
        }
        self.replacements
    }

    fn value_of(&self, value: ValueId) -> ValueId {
        match self.replacements.get(&value) {
            Some(replacement) => *replacement,
            None => value,
        }
    }

    fn number_of(&self, value: ValueId) -> ValueId {
        let value = self.value_of(value);
        match self.constant_numbers.get(&value) {
            Some(constant) => *constant,
            None => value,
        }
    }

    fn number(&mut self, cfg: &CfgBody, kind: &InstKind) {
        let Some(Numbered { dst, found }) = self.found(cfg, kind) else {
            return;
        };
        match found {
            Found::Const(word) => match self.table.get(&Expr::Const(word)) {
                Some(earlier) => {
                    let earlier = *earlier;
                    self.constant_numbers.insert(dst, earlier);
                }
                None => self.enter(Expr::Const(word), dst),
            },
            Found::Simplified(by) => self.replace(cfg, Replacement { replaced: dst, by }),
            Found::Expr(expr) => match self.table.get(&expr) {
                Some(earlier) => {
                    let by = *earlier;
                    self.replace(cfg, Replacement { replaced: dst, by });
                }
                None => self.enter(expr, dst),
            },
        }
    }

    fn enter(&mut self, expr: Expr, value: ValueId) {
        self.table.insert(expr.clone(), value);
        self.entered.push(expr);
    }

    fn replace(&mut self, cfg: &CfgBody, replacement: Replacement) {
        let Replacement { replaced, by } = replacement;
        assert_eq!(
            ty_of(cfg, replaced),
            ty_of(cfg, by),
            "{replaced:?} is replaced by {by:?}, which holds another type"
        );
        let previous = self.replacements.insert(replaced, by);
        assert!(
            previous.is_none(),
            "{replaced:?} is defined once and numbered once"
        );
    }

    fn operand(&self, value: ValueId) -> Option<Operand> {
        (!self.storages.contains(&value)).then(|| Operand {
            value: self.value_of(value),
            number: self.number_of(value),
        })
    }

    fn found(&self, cfg: &CfgBody, kind: &InstKind) -> Option<Numbered> {
        let numbered = |dst: ValueId, found: Found| Some(Numbered { dst, found });
        match kind {
            InstKind::Const { dst, .. } if !self.storages.contains(dst) => {
                numbered(*dst, Found::Const(*self.words.get(dst)?))
            }
            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => {
                let (left, right) = (self.operand(*left)?, self.operand(*right)?);
                let ty = ty_of(cfg, left.value);
                if let Some(value) = self.simplified(*op, [left, right], ty) {
                    return numbered(*dst, Found::Simplified(value));
                }
                let operands = Operands {
                    left: left.number,
                    right: right.number,
                };
                let operands = match commutes(*op, ty) {
                    true => operands.ordered(),
                    false => operands,
                };
                numbered(*dst, Found::Expr(Expr::Binary { op: *op, operands }))
            }
            InstKind::Cast { dst, src, to } => {
                let src = self.operand(*src)?.number;
                numbered(*dst, Found::Expr(Expr::Cast { to: *to, src }))
            }
            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let part = Part::Field {
                    field: *field,
                    rest: rest.clone(),
                };
                self.part(cfg, PartRead { dst: *dst, container: *object, part })
            }
            InstKind::ObjectGet { dst, object, key } => self.part(
                cfg,
                PartRead { dst: *dst, container: *object, part: Part::Key(*key) },
            ),
            InstKind::TupleIndex { dst, tuple, index } => self.part(
                cfg,
                PartRead { dst: *dst, container: *tuple, part: Part::TupleElement(*index) },
            ),
            InstKind::ArrayIndex { dst, array, index } => self.part(
                cfg,
                PartRead { dst: *dst, container: *array, part: Part::ArrayElement(*index) },
            ),
            _ => None,
        }
    }

    /// Only a scalar part of an aggregate held by value is numbered. A
    /// non-scalar part would have two owners once two reads were one.
    fn part(&self, cfg: &CfgBody, read: PartRead) -> Option<Numbered> {
        let PartRead { dst, container, part } = read;
        if !is_scalar(ty_of(cfg, dst)) {
            return None;
        }
        let container = self.operand(container)?.number;
        let held_by_value = match (&part, ty_of(cfg, container)) {
            (Part::Field { .. } | Part::Key(_), Ty::Object(_)) => true,
            (Part::TupleElement(_), Ty::Tuple(_)) => true,
            (Part::ArrayElement(_), Ty::Array(..)) => true,
            _ => false,
        };
        held_by_value.then_some(Numbered {
            dst,
            found: Found::Expr(Expr::Part { container, part }),
        })
    }

    /// Only integer identities are simplified. A float `x + 0.0` is `0.0`
    /// where `x` is `-0.0`, and `x * 0.0` is NaN where `x` is infinite.
    ///
    /// Each identity holds at either `Overflow`: `x + 0`, `x - 0`, `x * 1`
    /// and `x * 0` fit every width, so a trapping one never traps.
    ///
    /// The result is the operand's value, not its number, so an `x * 0`
    /// reads the `0` written beside it rather than an equal one above.
    fn simplified(&self, op: BinOp, operands: [Operand; 2], ty: &Ty) -> Option<ValueId> {
        let Ty::Int(_) = ty else {
            return None;
        };
        let [left, right] = operands;
        let int = |operand: Operand| {
            self.words
                .get(&operand.number)
                .and_then(|word| word.int_value())
        };
        if matches!(op, BinOp::Min | BinOp::Max) && left.number == right.number {
            return Some(left.value);
        }
        match (op, int(left), int(right)) {
            (BinOp::Add(_) | BinOp::Sub(_), _, Some(0)) | (BinOp::Mul(_), _, Some(1)) => {
                Some(left.value)
            }
            (BinOp::Add(_), Some(0), _) | (BinOp::Mul(_), Some(1), _) => Some(right.value),
            (BinOp::Mul(_), Some(0), _) => Some(left.value),
            (BinOp::Mul(_), _, Some(0)) => Some(right.value),
            _ => None,
        }
    }
}
