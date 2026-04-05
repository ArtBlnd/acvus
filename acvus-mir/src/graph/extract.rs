//! Phase 0: Extract
//!
//! Parse source ASTs and cache them for later phases.
//! Context dependency tracking is handled by the infer phase (typeck effect propagation),
//! not here — callee effects create transitive context deps that AST-level analysis cannot see.

use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use super::types::*;

// ── Phase 0 output ──────────────────────────────────────────────────

/// Phase 0 output: parsed ASTs for all local functions.
#[derive(Debug)]
pub struct ExtractResult {
    /// Parsed ASTs cached for later phases (avoid re-parsing).
    pub parsed: FxHashMap<QualifiedRef, ParsedSource>,
}

/// Cached parsed AST.
#[derive(Debug)]
pub enum ParsedSource {
    Script(acvus_ast::Script),
    Template(acvus_ast::Template),
}

// ── Extraction ─────────────────────────────────────────────────────

/// Run Phase 0: parse and cache ASTs for all local functions.
pub fn extract(interner: &Interner, graph: &CompilationGraph) -> ExtractResult {
    let mut parsed = FxHashMap::default();

    for func in graph.functions.iter() {
        if let Some(parsed_source) = extract_one(interner, func) {
            parsed.insert(func.qref, parsed_source);
        }
    }

    ExtractResult { parsed }
}

/// Parse a single local function. Returns None for Extern functions.
pub fn extract_one(_interner: &Interner, func: &Function) -> Option<ParsedSource> {
    match &func.kind {
        FnKind::Local(ast) => match ast {
            ParsedAst::Script(script) => {
                let _ = acvus_ast::extract_script_context_refs(script);
                Some(ParsedSource::Script(script.clone()))
            }
            ParsedAst::Template(template) => {
                let _ = acvus_ast::extract_template_context_refs(template);
                Some(ParsedSource::Template(template.clone()))
            }
        },
        FnKind::Extern => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::{Freeze, Interner};

    fn make_graph(interner: &Interner, source: &str) -> (CompilationGraph, QualifiedRef) {
        let fn_qref = QualifiedRef::root(interner.intern("test_unit"));
        let graph = CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref: fn_qref,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                    hint: None,
                },
            }]),
            contexts: Freeze::new(vec![]),
        };
        (graph, fn_qref)
    }

    #[test]
    fn extract_caches_parsed_ast() {
        let i = Interner::new();
        let (graph, uid) = make_graph(&i, "1 + 2");
        let result = extract(&i, &graph);
        assert!(
            result.parsed.contains_key(&uid),
            "parsed AST should be cached"
        );
    }

    #[test]
    fn extract_skips_extern() {
        let i = Interner::new();
        let qref = QualifiedRef::root(i.intern("ext"));
        let graph = CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref,
                kind: FnKind::Extern,
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                    hint: None,
                },
            }]),
            contexts: Freeze::new(vec![]),
        };
        let result = extract(&i, &graph);
        assert!(result.parsed.is_empty());
    }
}
