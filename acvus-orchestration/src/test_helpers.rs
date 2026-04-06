//! Test helpers: end-to-end compilation of Namespace specs.
//!
//! Not a public API — will be replaced when the real API is designed.

pub mod compile {
    use acvus_mir::graph::infer::InferResult;
    use acvus_mir::graph::{
        CompilationGraph, Function, QualifiedRef, extract, infer, lower as graph_lower,
    };
    use acvus_mir::ir::MirModule;
    use acvus_utils::{Freeze, Interner};
    use rustc_hash::FxHashMap;

    use crate::lower::{self, FieldError, SpanMap};
    use crate::spec::Namespace;

    /// End-to-end compilation result.
    pub struct CompileResult {
        /// Per-function MIR modules (only for Complete functions).
        pub modules: FxHashMap<QualifiedRef, MirModule>,
        /// Spec-level field errors (parse errors from inline content).
        pub field_errors: Vec<FieldError>,
        /// Span mapping for type error → spec field resolution.
        pub span_map: SpanMap,
        /// Infer result (for inspecting Complete/Incomplete outcomes).
        pub infer_result: InferResult,
        /// The lowered compilation graph (for inspecting functions).
        pub graph: CompilationGraph,
    }

    impl CompileResult {
        pub fn has_field_errors(&self) -> bool {
            !self.field_errors.is_empty()
        }

        pub fn has_infer_errors(&self) -> bool {
            self.infer_result.has_errors()
        }

        pub fn module(&self, id: QualifiedRef) -> Option<&MirModule> {
            self.modules.get(&id)
        }

        /// Find function by name.
        pub fn function_id(&self, interner: &Interner, name: &str) -> Option<QualifiedRef> {
            self.graph
                .functions
                .iter()
                .find(|f| interner.resolve(f.qref.name) == name)
                .map(|f| f.qref)
        }

        /// Whether a function compiled successfully (Complete + MIR produced).
        pub fn is_complete(&self, interner: &Interner, name: &str) -> bool {
            self.function_id(interner, name)
                .and_then(|id| self.module(id))
                .is_some()
        }

        /// Collect field error fields for a given item name.
        pub fn field_errors_for(&self, item_name: &str) -> Vec<&str> {
            self.field_errors
                .iter()
                .filter(|e| e.item_name == item_name)
                .map(|e| e.field.as_str())
                .collect()
        }

        /// Collect all span map origins as (item_name_or_display, field) tuples.
        pub fn span_origins(&self) -> Vec<(&str, &str)> {
            use crate::lower::SpecOrigin;
            self.span_map
                .entries
                .iter()
                .map(|e| match &e.origin {
                    SpecOrigin::LlmField { llm_name, field } => (llm_name.as_str(), field.as_str()),
                    SpecOrigin::DisplayField {
                        display_name,
                        field,
                    } => (display_name.as_str(), field.as_str()),
                })
                .collect()
        }
    }

    /// Compile a Namespace spec end-to-end: lower → extract → infer → lower to MIR.
    pub fn compile_namespace(
        interner: &Interner,
        ns: &Namespace,
        extern_fns: &[Function],
    ) -> CompileResult {
        // Phase 0: Spec → CompilationGraph
        let lowered = lower::lower_namespace(interner, ns, extern_fns);

        // Phase 1: Extract
        let ext = extract::extract(interner, &lowered.graph);

        // Phase 2: Infer (with policy + effect constraint validation)
        let inf = infer::infer(
            interner,
            &lowered.graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
            &FxHashMap::default(),
        );

        // Phase 3: Lower to MIR
        let mir_result =
            graph_lower::lower(interner, &lowered.graph, &ext, &inf, &FxHashMap::default());

        CompileResult {
            modules: mir_result.modules,
            field_errors: lowered.field_errors,
            span_map: lowered.span_map,
            infer_result: inf,
            graph: lowered.graph,
        }
    }
}
