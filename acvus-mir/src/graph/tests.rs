use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::graph::*;
use crate::ty::Ty;

    fn simple_name_to_id(interner: &Interner, pairs: &[(&str, ContextId)]) -> FxHashMap<acvus_utils::Astr, ContextId> {
        pairs.iter().map(|(name, id)| (interner.intern(name), *id)).collect()
    }

    /// Single script unit, no dependencies. Should resolve and produce Int.
    #[test]
    fn single_unit_resolves() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("1 + 2"),
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.resolutions.contains_key(&UnitId(0)));
        assert_eq!(result.resolutions[&UnitId(0)].tail_ty, Ty::Int);
    }

    /// Two units in DAG: unit 0 produces String, unit 1 references it via @data.
    #[test]
    fn dag_two_units() {
        let interner = Interner::new();
        let ctx_data = ContextId(0);

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("\"hello\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@data"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[("data", ctx_data)]),
                output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![ContextBinding {
                    id: ctx_data,
                    source: ContextSource::Derived(UnitId(0), TypeTransform::Identity),
                    constraint: None,
                }],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.resolutions[&UnitId(1)].tail_ty, Ty::String);
    }

    /// ExternDecl: declares output_ty, validates input type.
    #[test]
    fn extern_decl_output_type() {
        let interner = Interner::new();
        let ctx_raw = ContextId(0);

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("\"msg\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
            ],
            externs: vec![ExternDecl {
                id: UnitId(1),
                inputs: vec![(UnitId(0), Ty::String)],
                output_ty: Ty::Int, // extern produces Int
            }],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![ContextBinding {
                    id: ctx_raw,
                    source: ContextSource::Derived(UnitId(1), TypeTransform::Identity),
                    constraint: None,
                }],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // extern's output should be Int
        assert_eq!(result.unit_outputs[&UnitId(1)], Ty::Int);
    }

    /// ExternDecl input type mismatch → error.
    #[test]
    fn extern_decl_input_mismatch() {
        let interner = Interner::new();

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("123"), // produces Int
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
            ],
            externs: vec![ExternDecl {
                id: UnitId(1),
                inputs: vec![(UnitId(0), Ty::String)], // expects String!
                output_ty: Ty::Int,
            }],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(!result.errors.is_empty(), "should have type mismatch error");
    }

    /// External context variable (dynamic) with known type.
    #[test]
    fn external_known_type() {
        let interner = Interner::new();
        let ctx_input = ContextId(0);

        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("@input"),
                kind: SourceKind::Script,
                name_to_id: simple_name_to_id(&interner, &[("input", ctx_input)]),
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::from_iter([(ctx_input, ExternalType::Known(Ty::String))]),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.resolutions[&UnitId(0)].tail_ty, Ty::String);
    }

    // ── Soundness tests ─────────────────────────────────────────────
    // 잘못된 프로그램이 거부되는가?

    /// Undefined context → error.
    #[test]
    fn soundness_undefined_context_rejected() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("@unknown"),
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(), // @unknown은 name_to_id에 없음
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(!result.errors.is_empty(), "undefined context should produce error");
    }

    /// Type mismatch in expression → error.
    #[test]
    fn soundness_type_mismatch_rejected() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("1 + \"hello\""), // Int + String = error
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(!result.errors.is_empty(), "type mismatch should produce error");
    }

    /// Unresolved type variable (empty list without hint) → error.
    #[test]
    fn soundness_ambiguous_type_rejected() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("[]"), // no hint → ambiguous
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(!result.errors.is_empty(), "ambiguous type should be rejected");
    }

    /// Parse error → error.
    #[test]
    fn soundness_parse_error_rejected() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("1 +"), // incomplete expression
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(!result.errors.is_empty(), "parse error should be reported");
    }

    // ── Completeness tests ──────────────────────────────────────────
    // 올바른 프로그램이 수용되는가?

    /// SCC: two scripts share a ScopeLocal. Script 1 = `[]`, Script 2 provides elem type via @data.
    /// This is the init/bind pattern — `[]` alone is ambiguous, but bind resolves elem type.
    #[test]
    fn completeness_scc_resolves_ambiguous_list() {
        let interner = Interner::new();
        let ctx_self = ContextId(0);
        let ctx_data = ContextId(1);

        let graph = CompilationGraph {
            units: vec![
                // "init" — empty list, elem type unknown. Output → @self.
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("[]"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: Some(ctx_self),
                },
                // "bind" — appends @data to @self, resolving elem type. Output → @self.
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@self | append(@data)"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[
                        ("self", ctx_self),
                        ("data", ctx_data),
                    ]),
                    output_binding: Some(ctx_self),
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![
                    ContextBinding {
                        id: ctx_self,
                        source: ContextSource::ScopeLocal,
                        constraint: None, // no Sequence enforcement for this test
                    },
                ],
            }],
            externals: FxHashMap::from_iter([(ctx_data, ExternalType::Known(Ty::String))]),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        // Both units should compile without errors.
        assert!(result.errors.is_empty(), "SCC should resolve: {:?}", result.errors);
        // init's output should have resolved element type (String).
        assert!(result.resolutions.contains_key(&UnitId(0)));
        assert!(result.resolutions.contains_key(&UnitId(1)));
    }

    /// Valid chain: body produces String → bind appends to @self → correct.
    #[test]
    fn completeness_dag_chain_valid() {
        let interner = Interner::new();
        let ctx_raw = ContextId(0);

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("\"hello\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@raw"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[("raw", ctx_raw)]),
                output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![ContextBinding {
                    id: ctx_raw,
                    source: ContextSource::Derived(UnitId(0), TypeTransform::Identity),
                    constraint: None,
                }],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "valid chain should succeed: {:?}", result.errors);
        assert_eq!(result.resolutions[&UnitId(1)].tail_ty, Ty::String);
    }

    /// Template unit compiles successfully.
    #[test]
    fn completeness_template_unit() {
        let interner = Interner::new();
        let ctx_name = ContextId(0);

        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("hello {{ @name }}"),
                kind: SourceKind::Template,
                name_to_id: simple_name_to_id(&interner, &[("name", ctx_name)]),
                output_binding: None,
            }],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::from_iter([(ctx_name, ExternalType::Known(Ty::String))]),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "template should compile: {:?}", result.errors);
    }

    /// ElemOf transform: @item = elem_of(expr output).
    #[test]
    fn completeness_elem_of_transform() {
        let interner = Interner::new();
        let ctx_expr_output = ContextId(0); // unit 0's output (List<Int>)
        let ctx_item = ContextId(1);        // elem_of → Int

        let graph = CompilationGraph {
            units: vec![
                // expr: produces List<Int>
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("[1, 2, 3]"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                // transform: uses @item which should be Int (elem of List<Int>)
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@item + 10"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[("item", ctx_item)]),
                output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![
                    // unit 0's output has its own ContextId (Identity)
                    ContextBinding {
                        id: ctx_expr_output,
                        source: ContextSource::Derived(UnitId(0), TypeTransform::Identity),
                        constraint: None,
                    },
                    // @item = elem_of(unit 0's output)
                    ContextBinding {
                        id: ctx_item,
                        source: ContextSource::Derived(UnitId(0), TypeTransform::ElemOf),
                        constraint: None,
                    },
                ],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "ElemOf should work: {:?}", result.errors);
        assert_eq!(result.resolutions[&UnitId(1)].tail_ty, Ty::Int);
    }

    /// Soundness: ElemOf on non-collection → @item gets Error (poison).
    #[test]
    fn soundness_elem_of_non_collection() {
        let interner = Interner::new();
        let ctx_expr_output = ContextId(0); // unit 0's output (String)
        let ctx_item = ContextId(1);        // elem_of(String) → Error

        let graph = CompilationGraph {
            units: vec![
                // produces String, not a collection
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("\"hello\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                // uses @item = elem_of(String) → Error poison
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@item"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[("item", ctx_item)]),
                output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![
                    ContextBinding {
                        id: ctx_expr_output,
                        source: ContextSource::Derived(UnitId(0), TypeTransform::Identity),
                        constraint: None,
                    },
                    ContextBinding {
                        id: ctx_item,
                        source: ContextSource::Derived(UnitId(0), TypeTransform::ElemOf),
                        constraint: None,
                    },
                ],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        // @item = elem_of(String) = Error (poison).
        // Error unifies with anything, so unit 1 typechecks and tail_ty = Error.
        if let Some(res) = result.resolutions.get(&UnitId(1)) {
            assert!(res.tail_ty.is_error(), "tail should be Error, got {:?}", res.tail_ty);
        }
    }

    /// ScopeLocal with Sequence constraint: @self must be Sequence<β, O, Pure>.
    #[test]
    fn completeness_scope_local_with_constraint() {
        let interner = Interner::new();
        let ctx_self = ContextId(0);
        let ctx_data = ContextId(1);

        use crate::ty::{Effect, TySubst};
        // Build the constraint: Sequence<fresh_var, fresh_origin, Pure>
        let mut constraint_subst = TySubst::new();
        let beta = constraint_subst.fresh_var();
        let origin = constraint_subst.fresh_origin();
        let constraint = Ty::Sequence(Box::new(beta), origin, Effect::Pure);

        let graph = CompilationGraph {
            units: vec![
                // init: [] (empty deque). Output → @self.
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("[]"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: Some(ctx_self),
                },
                // bind: @self | chain([@data]). Output → @self.
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@self | chain([@data])"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[
                        ("self", ctx_self),
                        ("data", ctx_data),
                    ]),
                    output_binding: Some(ctx_self),
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![
                    ContextBinding {
                        id: ctx_self,
                        source: ContextSource::ScopeLocal,
                        constraint: Some(constraint),
                    },
                ],
            }],
            externals: FxHashMap::from_iter([(ctx_data, ExternalType::Known(Ty::String))]),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "Sequence constraint should work: {:?}", result.errors);
        // @self should be Sequence<String, O, Pure>
        if let Some(self_ty) = result.resolved_types.get(&ctx_self) {
            assert!(
                matches!(self_ty, Ty::Sequence(_, _, _)),
                "@self should be Sequence, got {:?}", self_ty
            );
        }
    }

    /// Errors are collected (not fail-fast) — multiple units can have errors.
    #[test]
    fn error_collection_not_fail_fast() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("1 + \"a\""), // type error
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("true + 1"), // type error
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        // Both errors should be collected, not just the first.
        assert!(result.errors.len() >= 2, "should collect errors from both units, got {}", result.errors.len());
    }

    // ── output_binding tests ────────────────────────────────────────

    /// output_binding connects unit output to ScopeLocal — trivial SCC with single unit.
    #[test]
    fn output_binding_trivial_scc() {
        let interner = Interner::new();
        let ctx_self = ContextId(0);

        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("[1, 2, 3]"),
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: Some(ctx_self),
            }],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0)],
                bindings: vec![ContextBinding {
                    id: ctx_self,
                    source: ContextSource::ScopeLocal,
                    constraint: None,
                }],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // @self should be resolved to the unit's output type
        let self_ty = result.resolved_types.get(&ctx_self);
        assert!(self_ty.is_some(), "@self should be resolved");
    }

    /// unit_outputs: each unit's output is tracked independently.
    #[test]
    fn unit_outputs_tracked() {
        let interner = Interner::new();

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("42"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("\"hello\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.unit_outputs[&UnitId(0)], Ty::Int);
        assert_eq!(result.unit_outputs[&UnitId(1)], Ty::String);
    }

    /// ExternDecl with covariant input: Deque ≤ Sequence should be accepted.
    #[test]
    fn extern_decl_covariant_input() {
        let interner = Interner::new();
        use crate::ty::{Effect, Origin};

        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("[1, 2, 3]"), // produces Deque<Int, O>
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: None,
            }],
            externs: vec![ExternDecl {
                id: UnitId(1),
                // Expects Iterator<Int, Pure> — Deque ≤ Iterator via coercion
                inputs: vec![(UnitId(0), Ty::Iterator(Box::new(Ty::Int), Effect::Pure))],
                output_ty: Ty::String,
            }],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        // Deque<Int, O> should coerce to Iterator<Int, Pure> — no error.
        assert!(result.errors.is_empty(), "covariant input should be accepted: {:?}", result.errors);
    }

    /// Multiple ScopeLocals: two independent scope locals in the same scope.
    #[test]
    fn multiple_scope_locals() {
        let interner = Interner::new();
        let ctx_a = ContextId(0);
        let ctx_b = ContextId(1);

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("42"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: Some(ctx_a),
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("\"hello\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: Some(ctx_b),
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![
                    ContextBinding { id: ctx_a, source: ContextSource::ScopeLocal, constraint: None },
                    ContextBinding { id: ctx_b, source: ContextSource::ScopeLocal, constraint: None },
                ],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "multiple scope locals should work: {:?}", result.errors);
    }

    // ── Phase 2: compile tests ──────────────────────────────────────

    /// compile() produces MirModule for each unit.
    #[test]
    fn compile_produces_mir_modules() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("1 + 2"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("hello world"),
                    kind: SourceKind::Template,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.compile(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.compiled.contains_key(&UnitId(0)), "script should be compiled");
        assert!(result.compiled.contains_key(&UnitId(1)), "template should be compiled");
        assert_eq!(result.compiled[&UnitId(0)].tail_ty, Ty::Int);
    }

    /// compile() skips ExternDecls (no MirModule) but registers their output.
    #[test]
    fn compile_extern_no_mir() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![CompilationUnit {
                id: UnitId(0),
                source: interner.intern("\"msg\""),
                kind: SourceKind::Script,
                name_to_id: FxHashMap::default(),
                output_binding: None,
            }],
            externs: vec![ExternDecl {
                id: UnitId(1),
                inputs: vec![(UnitId(0), Ty::String)],
                output_ty: Ty::Int,
            }],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.compile(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // Unit 0 has compiled MIR.
        assert!(result.compiled.contains_key(&UnitId(0)));
        // Extern (Unit 1) has no compiled MIR but has output type.
        assert!(!result.compiled.contains_key(&UnitId(1)));
        assert_eq!(result.unit_outputs[&UnitId(1)], Ty::Int);
    }

    /// compile() with SCC: both units produce MIR.
    #[test]
    fn compile_scc_produces_mir() {
        let interner = Interner::new();
        let ctx_self = ContextId(0);
        let ctx_data = ContextId(1);

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("[]"),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: Some(ctx_self),
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@self | append(@data)"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[
                        ("self", ctx_self),
                        ("data", ctx_data),
                    ]),
                    output_binding: Some(ctx_self),
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0), UnitId(1)],
                bindings: vec![ContextBinding {
                    id: ctx_self,
                    source: ContextSource::ScopeLocal,
                    constraint: None,
                }],
            }],
            externals: FxHashMap::from_iter([(ctx_data, ExternalType::Known(Ty::String))]),
            id_table: ContextIdTable::new(),
        };

        let result = graph.compile(&interner);
        assert!(result.errors.is_empty(), "SCC compile errors: {:?}", result.errors);
        assert!(result.compiled.contains_key(&UnitId(0)));
        assert!(result.compiled.contains_key(&UnitId(1)));
    }

    // ── Edge case / regression tests ────────────────────────────────

    /// Empty graph resolves without errors.
    #[test]
    fn empty_graph() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty());
        assert!(result.resolutions.is_empty());
    }

    /// Unit outside any scope can still reference a binding (regression for #3).
    #[test]
    fn unit_outside_scope_references_binding() {
        let interner = Interner::new();
        let ctx_data = ContextId(0);

        let graph = CompilationGraph {
            units: vec![
                // Unit 0: in a scope, produces String.
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("\"hello\""),
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                // Unit 1: NOT in any scope, but references @data → Unit 0's output.
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("@data"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[("data", ctx_data)]),
                    output_binding: None,
                },
            ],
            externs: vec![],
            // Only unit 0 is in the scope. Unit 1 is outside.
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0)],
                bindings: vec![ContextBinding {
                    id: ctx_data,
                    source: ContextSource::Derived(UnitId(0), TypeTransform::Identity),
                    constraint: None,
                }],
            }],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "outside-scope unit should work: {:?}", result.errors);
        assert_eq!(result.resolutions[&UnitId(1)].tail_ty, Ty::String);
    }

    /// Object constraint instantiation: Var inside Object fields gets fresh vars.
    #[test]
    fn constraint_with_object_type() {
        let interner = Interner::new();
        let ctx_self = ContextId(0);
        let ctx_data = ContextId(1);

        use crate::ty::TySubst;
        let mut cs = TySubst::new();
        let field_ty = cs.fresh_var();
        let constraint = Ty::Object(
            FxHashMap::from_iter([(interner.intern("value"), field_ty)]),
        );

        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("{value: @data,}"),
                    kind: SourceKind::Script,
                    name_to_id: simple_name_to_id(&interner, &[("data", ctx_data)]),
                    output_binding: Some(ctx_self),
                },
            ],
            externs: vec![],
            scopes: vec![Scope {
                id: ScopeId(0),
                units: vec![UnitId(0)],
                bindings: vec![ContextBinding {
                    id: ctx_self,
                    source: ContextSource::ScopeLocal,
                    constraint: Some(constraint),
                }],
            }],
            externals: FxHashMap::from_iter([(ctx_data, ExternalType::Known(Ty::Int))]),
            id_table: ContextIdTable::new(),
        };

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "object constraint should work: {:?}", result.errors);
        // @self should be Object with value: Int
        if let Some(self_ty) = result.resolved_types.get(&ctx_self) {
            assert!(matches!(self_ty, Ty::Object(_)), "@self should be Object, got {:?}", self_ty);
        }
    }

    /// compile() skips units with typecheck errors but still compiles others.
    #[test]
    fn compile_skips_errored_units() {
        let interner = Interner::new();
        let graph = CompilationGraph {
            units: vec![
                CompilationUnit {
                    id: UnitId(0),
                    source: interner.intern("1 + \"bad\""), // type error
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
                CompilationUnit {
                    id: UnitId(1),
                    source: interner.intern("42"), // fine
                    kind: SourceKind::Script,
                    name_to_id: FxHashMap::default(),
                    output_binding: None,
                },
            ],
            externs: vec![],
            scopes: vec![],
            externals: FxHashMap::default(),
            id_table: ContextIdTable::new(),
        };

        let result = graph.compile(&interner);
        // Unit 0 has errors — not compiled.
        assert!(!result.compiled.contains_key(&UnitId(0)));
        // Unit 1 is fine — compiled.
        assert!(result.compiled.contains_key(&UnitId(1)));
        assert_eq!(result.compiled[&UnitId(1)].tail_ty, Ty::Int);
        // Errors collected.
        assert!(!result.errors.is_empty());
    }
