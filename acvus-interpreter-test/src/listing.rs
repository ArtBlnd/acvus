//! A prepared body's blocks, read back as names (RFC-0052 §1, §3).

use std::sync::Arc;

use acvus_extern::Registry;
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::code::{Body, Code, Named, Op, Prepared, Shape, Slot, Where};
use acvus_interpreter::{PrepareCtx, prepare_module};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

use crate::{Context, ParsedAst, compile_script_mode, compile_source_with_externs, split_context};

pub struct BlockListing {
    pub ops: Vec<String>,
    pub end: String,
    pub regions: Vec<RegionListing>,
}

/// One region operation: its name, and the parts it holds.
pub struct RegionListing {
    pub name: String,
    pub owns: Vec<PartListing>,
}

pub struct PartListing {
    pub part: String,
    pub ops: Vec<String>,
    pub regions: Vec<RegionListing>,
    pub leaves_with: usize,
}

/// The operation's type with every module path dropped, inside its generic
/// arguments as well, so that `arith::Lt<i64, place::Slot, place::Slot,
/// place::R0>` reads as `Lt<i64, Slot, Slot, R0>`.
pub fn last_path_segment(of: &dyn Named) -> String {
    let full = of.name();
    let mut shortened = String::with_capacity(full.len());
    let mut path = String::new();
    for c in full.chars() {
        match c.is_alphanumeric() || c == '_' || c == ':' {
            true => path.push(c),
            false => {
                shortened.push_str(tail(&path));
                path.clear();
                shortened.push(c);
            }
        }
    }
    shortened.push_str(tail(&path));
    shortened
}

fn tail(path: &str) -> &str {
    match path.rsplit_once("::") {
        Some((_, last)) => last,
        None => path,
    }
}

/// A chain read back as the operations it holds and the one node that ends
/// it: the successor field is what is walked, so a listing and the machine
/// read the same chain.
fn walk(head: &dyn Op) -> (Vec<&dyn Op>, &dyn Op) {
    let mut ops = Vec::new();
    let mut at = head;
    while let Some(next) = at.successor() {
        ops.push(at);
        at = next;
    }
    (ops, at)
}

fn names_of(ops: &[&dyn Op]) -> Vec<String> {
    ops.iter().map(|op| last_path_segment(*op)).collect()
}

/// A region is the one operation that answers `owns`, so this is every
/// region of an operation list, in order.
fn regions_of(ops: &[&dyn Op]) -> Vec<RegionListing> {
    ops.iter()
        .filter(|op| !op.owns().is_empty())
        .map(|op| RegionListing {
            name: last_path_segment(*op),
            owns: op
                .owns()
                .into_iter()
                .map(|owned| {
                    let (part_ops, _end) = walk(owned.head);
                    PartListing {
                        part: owned.part.to_string(),
                        ops: names_of(&part_ops),
                        regions: regions_of(&part_ops),
                        leaves_with: moves_in(&part_ops),
                    }
                })
                .collect(),
        })
        .collect()
}

pub fn listing(heads: &[Box<dyn Op>]) -> Vec<BlockListing> {
    heads
        .iter()
        .map(|head| {
            let (ops, end) = walk(head.as_ref());
            BlockListing {
                ops: names_of(&ops),
                end: last_path_segment(end),
                regions: regions_of(&ops),
            }
        })
        .collect()
}

/// The moves a part leaves with: under RFC-0052 rule 1 they are `Mov`
/// operations of the part itself, not a list a terminator walks.
fn moves_in(ops: &[&dyn Op]) -> usize {
    names_of(ops)
        .iter()
        .filter(|name| name.starts_with("Mov<"))
        .count()
}

impl BlockListing {
    pub fn terminators_depth_first(&self, into: &mut Vec<String>) {
        into.push(self.end.clone());
    }
}

impl RegionListing {
    pub fn part(&self, part: &str) -> Option<&PartListing> {
        self.owns.iter().find(|owned| owned.part == part)
    }
}

pub fn terminators_depth_first(blocks: &[BlockListing]) -> Vec<String> {
    let mut names = Vec::new();
    for block in blocks {
        block.terminators_depth_first(&mut names);
    }
    names
}

/// Every region of this name, a nested one included, in the order the
/// operations hold them.
pub fn regions_named<'b>(blocks: &'b [BlockListing], name: &str) -> Vec<&'b RegionListing> {
    let mut found = Vec::new();
    for block in blocks {
        collect_regions(&block.regions, name, &mut found);
    }
    found
}

/// The operation's family: its type name without the arguments its
/// specialization added, which is how `asm_probe` names one too.
pub fn family_of(name: &str) -> &str {
    match name.split_once('<') {
        Some((head, _)) => head,
        None => name,
    }
}

fn collect_regions<'b>(
    regions: &'b [RegionListing],
    name: &str,
    found: &mut Vec<&'b RegionListing>,
) {
    for region in regions {
        for part in &region.owns {
            collect_regions(&part.regions, name, found);
        }
        if family_of(&region.name) == name {
            found.push(region);
        }
    }
}

pub fn ops_of(blocks: &[BlockListing]) -> Vec<String> {
    blocks.iter().flat_map(|b| b.ops.iter().cloned()).collect()
}

pub fn prepared_script_with_externs(
    interner: &Interner,
    source: &str,
    context: Context,
    registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
) -> Arc<Prepared> {
    let (context_types, _snapshot) = split_context(interner, context);
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let cr = compile_source_with_externs(interner, ast, &context_types, registries, ret);
    let ctx = PrepareCtx {
        interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    Arc::new(prepare_module(module, &ctx))
}

pub fn script_listing_with_externs(
    interner: &Interner,
    source: &str,
    context: Context,
    registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
) -> Vec<BlockListing> {
    listing(
        &main_body(&prepared_script_with_externs(
            interner, source, context, registries, ret,
        ))
        .heads,
    )
}

pub fn prepared_script(
    interner: &Interner,
    source: &str,
    context: Context,
    ret: Ty,
) -> Arc<Prepared> {
    let (context_types, _snapshot) = split_context(interner, context);
    let cr = compile_script_mode(interner, source, &context_types, ret);
    let ctx = PrepareCtx {
        interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    Arc::new(prepare_module(module, &ctx))
}

/// # Panics
/// The entry module's `main` is one chain, which a body-shape test has
/// nothing to read.
pub fn main_body(prepared: &Prepared) -> &Body {
    let Code::Body(body) = prepared.main.as_ref() else {
        panic!("the entry module's main is a body, not a one-chain expression")
    };
    body
}

/// Every chain operation of a body, those inside its superinstructions
/// included.
pub fn chains_of(code: &Code) -> Vec<ChainShape> {
    let Code::Body(body) = code else {
        return Vec::new();
    };
    let mut found = Vec::new();
    collect_chains(&body.heads, &mut found);
    found
}

/// One chain operation as a shape test reads it: where it writes, the tree
/// it walks, and the frame offsets of its leaves. No `dst` means the root's
/// word rides in the argument register (RFC-0052 rule 5).
pub struct ChainShape {
    pub dst: Option<Slot>,
    pub shape: Shape,
    pub leaves: Vec<u16>,
}

fn collect_chains(heads: &[Box<dyn Op>], found: &mut Vec<ChainShape>) {
    for head in heads {
        let (ops, _end) = walk(head.as_ref());
        collect_chains_of(&ops, found);
    }
}

fn collect_chains_of(ops: &[&dyn Op], found: &mut Vec<ChainShape>) {
    for op in ops {
        if let Some(probe) = op.chain() {
            found.push(ChainShape {
                dst: match probe.dst {
                    Where::Frame(off) => Some(
                        u16::try_from(off.index())
                            .expect("a register index fits the Slot its body was prepared with"),
                    ),
                    Where::Register => None,
                },
                shape: probe.plan.shape,
                leaves: probe.plan.leaves[..probe.plan.shape.leaves()].to_vec(),
            });
        }
        for owned in op.owns() {
            let (part_ops, _end) = walk(owned.head);
            collect_chains_of(&part_ops, found);
        }
    }
}

pub fn code_listing(code: &Code) -> Vec<BlockListing> {
    match code {
        Code::Body(body) => listing(&body.heads),
        Code::Expr(_) => Vec::new(),
    }
}

pub fn ops_of_anywhere(blocks: &[BlockListing]) -> Vec<String> {
    let mut found = Vec::new();
    collect_ops(blocks, &mut found);
    found
}

fn collect_ops(blocks: &[BlockListing], found: &mut Vec<String>) {
    for block in blocks {
        found.extend(block.ops.iter().cloned());
        for region in &block.regions {
            collect_part_ops(&region.owns, found);
        }
    }
}

fn collect_part_ops(parts: &[PartListing], found: &mut Vec<String>) {
    for part in parts {
        found.extend(part.ops.iter().cloned());
        for region in &part.regions {
            collect_part_ops(&region.owns, found);
        }
    }
}

pub fn script_listing(
    interner: &Interner,
    source: &str,
    context: Context,
    ret: Ty,
) -> Vec<BlockListing> {
    listing(&main_body(&prepared_script(interner, source, context, ret)).heads)
}
