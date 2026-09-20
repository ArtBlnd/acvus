//! A prepared body's blocks, read back as names (RFC-0052 §1, §3).

use std::fmt;

use serde::Serialize;

use crate::code::{Body, Code, Named, Op};

#[derive(Serialize)]
pub struct BlockListing {
    pub ops: Vec<String>,
    pub end: String,
    pub regions: Vec<RegionListing>,
}

/// One region operation: its name, and the parts it holds.
#[derive(Serialize)]
pub struct RegionListing {
    pub name: String,
    pub owns: Vec<PartListing>,
}

#[derive(Serialize)]
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
pub struct Chain<'o> {
    pub ops: Vec<&'o dyn Op>,
    pub end: &'o dyn Op,
}

pub fn walk(head: &dyn Op) -> Chain<'_> {
    let mut ops = Vec::new();
    let mut at = head;
    while let Some(next) = at.successor() {
        ops.push(at);
        at = next;
    }
    Chain { ops, end: at }
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
                    let part_ops = walk(owned.head).ops;
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
            let chain = walk(head.as_ref());
            BlockListing {
                ops: names_of(&chain.ops),
                end: last_path_segment(chain.end),
                regions: regions_of(&chain.ops),
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

pub fn code_listing(code: &Code) -> Vec<BlockListing> {
    match code {
        Code::Body(body) => body_listing(body),
        Code::Expr(_) => Vec::new(),
    }
}

pub fn body_listing(body: &Body) -> Vec<BlockListing> {
    listing(&body.heads)
}

/// One `L<n>` per chain head, its operations under it, and a region's parts
/// nested under the operation that owns them.
///
/// Obligation across artifacts: the oplist dumps of
/// `acvus-interpreter-test/benches/{programs,shapes,logs}.rs` print their
/// blocks in this shape, and `acvus ops` prints this one; a reader compares
/// the two by eye.
pub struct Text<'b> {
    blocks: &'b [BlockListing],
}

pub fn text(blocks: &[BlockListing]) -> Text<'_> {
    Text { blocks }
}

impl fmt::Display for Text<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, block) in self.blocks.iter().enumerate() {
            writeln!(f, "L{index}: {} ops, end {}", block.ops.len(), block.end)?;
            for op in &block.ops {
                writeln!(f, "  {op}")?;
            }
            write_regions(&block.regions, 2, f)?;
        }
        Ok(())
    }
}

/// `Frameless` is a prepared `Expr`: one chain the machine runs with no frame,
/// no block and no dispatch loop (RFC-0044, stage 4).
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeText {
    Blocks(Vec<BlockListing>),
    Frameless,
}

pub fn code_text(code: &Code) -> CodeText {
    match code {
        Code::Body(body) => body_text(body),
        Code::Expr(_) => CodeText::Frameless,
    }
}

pub fn body_text(body: &Body) -> CodeText {
    CodeText::Blocks(listing(&body.heads))
}

impl fmt::Display for CodeText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodeText::Blocks(blocks) => text(blocks).fmt(f),
            CodeText::Frameless => writeln!(f, "one frameless chain, no blocks"),
        }
    }
}

fn write_regions(
    regions: &[RegionListing],
    indent: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let pad = " ".repeat(indent);
    for region in regions {
        writeln!(f, "{pad}region {}", region.name)?;
        for part in &region.owns {
            write_part(part, indent + 2, f)?;
        }
    }
    Ok(())
}

fn write_part(part: &PartListing, indent: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let pad = " ".repeat(indent);
    writeln!(
        f,
        "{pad}{}: {} ops, leaves with {} moves",
        part.part,
        part.ops.len(),
        part.leaves_with
    )?;
    for op in &part.ops {
        writeln!(f, "{pad}  {op}")?;
    }
    write_regions(&part.regions, indent + 2, f)
}
