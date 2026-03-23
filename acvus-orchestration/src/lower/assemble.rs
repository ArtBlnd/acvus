//! Assembly: CompiledGraph + ExternFactory + UnitMeta → runtime Units (LowerResult).

use std::sync::Arc;

use acvus_mir::graph::{GraphError, Id};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::http::Fetch;
use crate::unit::{AssertNode, InterpreterUnit, Unit};

use super::{ExternFactory, LowerOutput, LowerResult, UnitMeta};

#[derive(Debug)]
pub enum AssembleError {
    /// Graph compilation produced errors (type errors, parse errors, etc.)
    Compile(Vec<GraphError>),
    /// No root entity found (empty graph).
    NoRoot,
    /// Multiple root entities found. Exactly 1 root required.
    MultipleRoots(Vec<Id>),
}

/// Phase 1+2 + Assembly: compile the graph, then create runtime Units.
///
/// Only spawnable entities (LocalUnit, ExternUnit) become runtime Units.
/// Context entities (provides, @raw, @self) are not spawnable — the resolver
/// resolves them via external resolver or turn_context, never via spawn.
pub fn assemble<F: Fetch + 'static>(
    output: LowerOutput,
    interner: &Interner,
    fetch: Arc<F>,
) -> Result<LowerResult, AssembleError> {
    // Phase 1+2: type resolution + MIR compilation.
    let id_table = output.graph.id_table.clone();
    let compiled = output.graph.compile(interner);

    let (compiled_units, _entity_types, errors, roots) = compiled.into_inner();

    if !errors.is_empty() {
        return Err(AssembleError::Compile(errors));
    }

    let entrypoint = match roots.len() {
        0 => return Err(AssembleError::NoRoot),
        1 => roots[0],
        _ => return Err(AssembleError::MultipleRoots(roots)),
    };

    let mut units: FxHashMap<Id, (Arc<dyn Unit>, UnitMeta)> = FxHashMap::default();
    let mut metas = output.unit_meta;

    // LocalUnit entities → InterpreterUnit.
    for (id, cu) in compiled_units.iter() {
        let unit: Arc<dyn Unit> = Arc::new(InterpreterUnit::new(cu.module.clone(), interner));
        let meta = metas.remove(id)
            .unwrap_or_else(|| panic!("no meta for compiled unit {id:?}"));
        units.insert(*id, (unit, meta));
    }

    // ExternUnit entities → provider-specific Units.
    for (id, factory) in output.extern_factories {
        let unit: Arc<dyn Unit> = match factory {
            ExternFactory::Assert { check_id, value_id, retry } => {
                Arc::new(AssertNode::new(check_id, value_id, retry))
            }
            ExternFactory::Init { storage_read_id, init_value_id } => {
                Arc::new(crate::unit::InitNode::new(storage_read_id, init_value_id))
            }
            ExternFactory::OpenAI(config) => {
                Arc::new(crate::unit::openai::OpenAIUnit::new(config, fetch.clone(), interner))
            }
            ExternFactory::Anthropic(config) => {
                Arc::new(crate::unit::anthropic::AnthropicUnit::new(config, fetch.clone(), interner))
            }
            ExternFactory::Google(config) => {
                Arc::new(crate::unit::google::GoogleUnit::new(config, fetch.clone(), interner))
            }
            ExternFactory::GoogleCache(_config) => {
                todo!("GoogleCache runtime unit not yet implemented")
            }
        };
        let meta = metas.remove(&id)
            .unwrap_or_else(|| panic!("no meta for extern unit {id:?}"));
        units.insert(id, (unit, meta));
    }

    Ok(LowerResult {
        units: Freeze::new(units),
        entrypoint,
        id_table,
    })
}
