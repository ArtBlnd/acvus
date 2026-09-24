//! A prepared body read back, over the sources a test compiles.
//!
//! The walk itself is `acvus_interpreter::listing`, which `acvus ops` prints
//! from; what stays here needs this crate's compile pipeline or its probe
//! types.

use std::sync::Arc;

use acvus_extern::Registry;
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::LeafRead;
use acvus_interpreter::code::{Body, Op, Prepared, Shape, Slot, Where};
use acvus_interpreter::{PrepareCtx, prepare_module};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

pub use acvus_interpreter::listing::{
    BlockListing, Chain, CodeText, PartListing, RegionListing, Text, code_listing, code_text,
    family_of, last_path_segment, listing, ops_of, ops_of_anywhere, regions_named,
    terminators_depth_first, text, walk,
};

use crate::{Context, ParsedAst, compile_script_mode, compile_source_with_externs, split_context};

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
        instances: &cr.instances,
        access: acvus_mir::graph::Access::Sync,
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
        instances: &cr.instances,
        access: acvus_mir::graph::Access::Sync,
    };
    let module = cr.modules.get(&cr.entry_qref).expect("the entry module");
    Arc::new(prepare_module(module, &ctx))
}

pub fn main_body(prepared: &Prepared) -> &Body {
    prepared.main.as_ref()
}

pub fn chains_of_body(body: &Body) -> Vec<ChainShape> {
    let mut found = Vec::new();
    collect_chains(&body.heads, &mut found);
    found
}

/// One chain operation as a shape test reads it: where it writes, the tree
/// it walks, the frame offsets of its leaves, and the type each leaf reads
/// its offset at (RFC-0049). No `dst` means the root's word rides in the
/// argument register (RFC-0052 rule 5).
pub struct ChainShape {
    pub dst: Option<Slot>,
    pub shape: Shape,
    pub leaves: Vec<u16>,
    pub reads: Vec<LeafRead>,
}

fn collect_chains(heads: &[Box<dyn Op>], found: &mut Vec<ChainShape>) {
    for head in heads {
        collect_chains_of(&walk(head.as_ref()).ops, found);
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
                reads: probe.plan.reads.leafwise()[..probe.plan.shape.leaves()].to_vec(),
            });
        }
        for owned in op.owns() {
            collect_chains_of(&walk(owned.head).ops, found);
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
