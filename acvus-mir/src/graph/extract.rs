//! Phase 0: Extract
//!
//! Two-step extraction:
//! 1. AST-level: parse source, extract context names.
//! 2. IR-level: build skeleton MIR with temp ids, run val_def to trace
//!    projection chains and determine which contexts are truly written to.

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::val_def::{ValDefMap, ValDefMapAnalysis};
use crate::ir::{InstKind, MirModule};
use crate::lower::Lowerer;
use crate::pass::AnalysisPass;

use super::types::*;

// ── Phase 0 output ──────────────────────────────────────────────────

/// Extracted information from a single function.
#[derive(Debug, Clone)]
pub struct FnRefs {
    /// Context names this function reads.
    pub context_reads: FxHashSet<QualifiedRef>,
    /// Context names this function writes to (traced through projection chains).
    pub context_writes: FxHashSet<QualifiedRef>,
}

/// Phase 0 output: extracted references for all functions.
#[derive(Debug)]
pub struct ExtractResult {
    pub fn_refs: FxHashMap<FunctionId, FnRefs>,
    /// Parsed ASTs cached for later phases (avoid re-parsing).
    pub parsed: FxHashMap<FunctionId, ParsedSource>,
}

/// Cached parsed AST.
#[derive(Debug)]
pub enum ParsedSource {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

// ── Per-function extraction ──────────────────────────────────────────

/// Extract information from a single local function.
/// Returns None if the function is not Local/LocalAst or fails to parse.
pub fn extract_one(interner: &Interner, func: &Function) -> Option<(FnRefs, ParsedSource)> {
    // Step 1: Parse AST and extract context names.
    let (context_names, parsed_source) = match &func.kind {
        FnKind::Local(source) => {
            let source_str = interner.resolve(source.source);
            match source.kind {
                SourceKind::Script => {
                    let script = acvus_ast::parse_script(interner, source_str).ok()?;
                    let names = acvus_ast::extract_script_context_refs(&script);
                    (names, ParsedSource::Script(script))
                }
                SourceKind::Template => {
                    let template = acvus_ast::parse(interner, source_str).ok()?;
                    let names = acvus_ast::extract_template_context_refs(&template);
                    (names, ParsedSource::Template(template))
                }
            }
        }
        FnKind::LocalAst(ast) => {
            match ast {
                ParsedAst::Script(script) => {
                    let names = acvus_ast::extract_script_context_refs(script);
                    (names, ParsedSource::Script(script.clone()))
                }
                ParsedAst::Template(template) => {
                    let names = acvus_ast::extract_template_context_refs(template);
                    (names, ParsedSource::Template(template.clone()))
                }
            }
        }
        FnKind::Extern => return None,
    };

    // Step 2: Build QualifiedRef→Ty mapping for skeleton MIR.
    let context_ids: FxHashMap<QualifiedRef, crate::ty::Ty> = context_names
        .iter()
        .map(|&qref| (qref, crate::ty::Ty::error()))
        .collect();
    let qref_to_name: FxHashMap<QualifiedRef, Astr> = context_names
        .iter()
        .map(|&qref| (qref, qref.name))
        .collect();

    // Step 3: Build skeleton MIR (empty type maps).
    let lowerer = Lowerer::new(
        interner,
        FxHashMap::default(),
        Vec::new(),
        Freeze::new(context_ids),
        Freeze::new(FxHashMap::default()),
    );
    let (module, _) = match &parsed_source {
        ParsedSource::Script(script) => lowerer.lower_script(script),
        ParsedSource::Template(template) => lowerer.lower_template(template),
    };

    // Step 4: Run val_def analysis and trace write chains.
    let refs = analyze_refs(&module, &qref_to_name);

    Some((refs, parsed_source))
}

// ── Batch extraction ────────────────────────────────────────────────

/// Run Phase 0 extraction on a compilation graph (batch).
pub fn extract(interner: &Interner, graph: &CompilationGraph) -> ExtractResult {
    let mut fn_refs = FxHashMap::default();
    let mut parsed = FxHashMap::default();

    for func in graph.functions.iter() {
        if let Some((refs, parsed_source)) = extract_one(interner, func) {
            fn_refs.insert(func.id, refs);
            parsed.insert(func.id, parsed_source);
        }
    }

    ExtractResult { fn_refs, parsed }
}

// ── IR analysis ─────────────────────────────────────────────────────

/// Analyze skeleton MIR to extract reads and writes with projection chain tracing.
fn analyze_refs(module: &MirModule, qref_to_name: &FxHashMap<QualifiedRef, Astr>) -> FnRefs {
    let val_def = ValDefMapAnalysis.run(module, ());
    let insts = &module.main.insts;

    let mut reads = FxHashSet::default();
    let mut writes = FxHashSet::default();

    for inst in insts {
        match &inst.kind {
            // Every ContextProject is a read.
            InstKind::ContextProject { ctx, .. } => {
                if qref_to_name.contains_key(ctx) {
                    reads.insert(*ctx);
                }
            }
            // ContextStore: trace dst back through projection chain to find root context.
            InstKind::ContextStore { dst, .. } => {
                if let Some(root_ref) = trace_projection_root(*dst, &val_def, insts)
                    && qref_to_name.contains_key(&root_ref)
                {
                    writes.insert(root_ref);
                }
            }
            _ => {}
        }
    }

    // Also walk closures.
    for closure in module.closures.values() {
        let closure_val_def = {
            // Build val_def for the closure body.
            let closure_module = MirModule {
                main: closure.clone(),
                closures: FxHashMap::default(),
            };
            ValDefMapAnalysis.run(&closure_module, ())
        };
        let closure_insts = &closure.insts;

        for inst in closure_insts {
            match &inst.kind {
                InstKind::ContextProject { ctx, .. } => {
                    if qref_to_name.contains_key(ctx) {
                        reads.insert(*ctx);
                    }
                }
                InstKind::ContextStore { dst, .. } => {
                    if let Some(root_ref) =
                        trace_projection_root(*dst, &closure_val_def, closure_insts)
                        && qref_to_name.contains_key(&root_ref)
                    {
                        writes.insert(root_ref);
                    }
                }
                _ => {}
            }
        }
    }

    FnRefs {
        context_reads: reads,
        context_writes: writes,
    }
}

/// Trace a ValueId back through the val_def chain to find the root ContextProject's id.
///
/// e.g., ContextStore { dst: v2, .. }
///   → v2 defined by FieldGet { dst: v2, object: v1, .. }
///   → v1 defined by ContextProject { dst: v1, id: ctx_id }
///   → root is ctx_id
fn trace_projection_root(
    val: crate::ir::ValueId,
    val_def: &ValDefMap,
    insts: &[crate::ir::Inst],
) -> Option<QualifiedRef> {
    let mut current = val;
    // Limit iterations to prevent infinite loops on malformed IR.
    for _ in 0..64 {
        let inst_idx = val_def.0.get(&current)?;
        let inst = insts.get(*inst_idx)?;
        match &inst.kind {
            InstKind::ContextProject { ctx, .. } => return Some(*ctx),
            InstKind::FieldGet { object, .. } => {
                current = *object;
            }
            // Can't trace further — not a projection chain.
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    fn make_graph(
        interner: &Interner,
        source: &str,
        ctx_names: &[&str],
    ) -> (CompilationGraph, FunctionId) {
        let fn_id = FunctionId::alloc();
        let contexts: Vec<Context> = ctx_names
            .iter()
            .map(|&name| Context {
                name: interner.intern(name),
                namespace: None,
                constraint: Constraint::Exact(crate::ty::Ty::Int),
            })
            .collect();
        let graph = CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(vec![Function {
                id: fn_id,
                name: interner.intern("test_unit"),
                namespace: None,
                kind: FnKind::Local(SourceCode {
                    name: interner.intern("test"),
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                },
            }]),
            contexts: Freeze::new(contexts),
        };
        (graph, fn_id)
    }

    // -- Completeness: valid reads detected --

    #[test]
    fn extract_detects_context_read() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "@x", &["x"]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(
            refs.context_reads
                .contains(&QualifiedRef::root(i.intern("x"))),
            "should detect @x read"
        );
    }

    #[test]
    fn extract_detects_multiple_reads() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "@a + @b", &["a", "b"]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(
            refs.context_reads
                .contains(&QualifiedRef::root(i.intern("a")))
        );
        assert!(
            refs.context_reads
                .contains(&QualifiedRef::root(i.intern("b")))
        );
    }

    // -- Completeness: direct writes detected --

    #[test]
    fn extract_detects_direct_write() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "@x = 42; @x", &["x"]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(
            refs.context_writes
                .contains(&QualifiedRef::root(i.intern("x"))),
            "should detect @x write"
        );
    }

    // -- Completeness: projection chain write detected --

    #[test]
    fn extract_detects_field_write_traces_to_root() {
        let i = Interner::new();
        // @obj.name = "new" should trace back to @obj as the write target.
        // But currently the AST doesn't support `@obj.name = ...` syntax directly.
        // Instead, test direct @x = ... which is the current supported form.
        let (graph, uid) = make_graph(&i, "@x = @x + 1; @x", &["x"]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(
            refs.context_writes
                .contains(&QualifiedRef::root(i.intern("x")))
        );
        // @x is also read (in @x + 1).
        assert!(
            refs.context_reads
                .contains(&QualifiedRef::root(i.intern("x")))
        );
    }

    // -- Soundness: no false writes --

    #[test]
    fn extract_read_only_no_writes() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "@x + 1", &["x"]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(
            refs.context_reads
                .contains(&QualifiedRef::root(i.intern("x")))
        );
        assert!(
            refs.context_writes.is_empty(),
            "read-only should have no writes"
        );
    }

    #[test]
    fn extract_no_contexts_empty() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "1 + 2", &[]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(refs.context_reads.is_empty());
        assert!(refs.context_writes.is_empty());
    }

    // -- Soundness: write to one context doesn't mark another --

    #[test]
    fn extract_write_does_not_leak_to_other_context() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "@x = 42; @y", &["x", "y"]);
        let result = extract(&i, &graph);
        let refs = &result.fn_refs[&uid];
        assert!(
            refs.context_writes
                .contains(&QualifiedRef::root(i.intern("x"))),
            "@x should be written"
        );
        assert!(
            !refs
                .context_writes
                .contains(&QualifiedRef::root(i.intern("y"))),
            "@y should NOT be written"
        );
    }

    // -- Parsed ASTs cached --

    #[test]
    fn extract_caches_parsed_ast() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "@x", &["x"]);
        let result = extract(&i, &graph);
        assert!(
            result.parsed.contains_key(&uid),
            "parsed AST should be cached"
        );
    }
}
