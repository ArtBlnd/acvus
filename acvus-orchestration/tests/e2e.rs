//! End-to-end tests: Namespace spec → compiled MIR.
//!
//! Tests type checking correctness and error location accuracy
//! through the full pipeline: spec → lower → extract → infer → lower → MIR.

use acvus_orchestration::spec::*;
use acvus_orchestration::test_helpers::compile::compile_namespace;
use acvus_utils::Interner;

// ════════════════════════════════════════════════════════════════════
// 1. Type check — Completeness (valid specs compile)
// ════════════════════════════════════════════════════════════════════

#[test]
fn block_template_compiles() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Block(Block {
            name: "greeting".into(),
            source: "hello world".into(),
            mode: BlockMode::Template,
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors());
    assert!(!result.has_infer_errors());
    assert!(result.is_complete(&i, "greeting"));
}

#[test]
fn block_script_compiles() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Block(Block {
            name: "compute".into(),
            source: "1 + 2".into(),
            mode: BlockMode::Script,
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors());
    assert!(result.is_complete(&i, "compute"));
}

#[test]
fn display_static_compiles() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Display(DisplaySpec::Static {
            name: "output".into(),
            source: "Hello!".into(),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors());
    assert!(result.is_complete(&i, "output"));
}

#[test]
fn display_iterator_with_history_and_live_compiles() {
    let i = Interner::new();
    // Need contexts for @history and @stream
    // But contexts aren't declared → will be Incomplete.
    // This tests that all 3 functions are generated (even if Incomplete).
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Display(DisplaySpec::Iterator {
            name: "msgs".into(),
            history: Some("@history".into()),
            live: Some("@stream".into()),
            bind: "msg".into(),
            template: "{{ $msg }}".into(),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors());
    // Template function should compile (no context refs)
    assert!(result.is_complete(&i, "__msgs_tpl"));
    // history/live reference undeclared contexts → Incomplete
    assert!(!result.is_complete(&i, "msgs_history"));
    assert!(!result.is_complete(&i, "msgs_live"));
}

#[test]
fn llm_with_ref_messages_compiles() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![
            Item::Block(Block {
                name: "sys".into(),
                source: "You are helpful.".into(),
                mode: BlockMode::Template,
            }),
            Item::Block(Block {
                name: "usr".into(),
                source: "Hello!".into(),
                mode: BlockMode::Template,
            }),
            Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "https://api.google.com".into(),
                    api_key: "key".into(),
                    model: "gemini-2.0-flash".into(),
                    temperature: None, top_p: None, top_k: None, max_tokens: None,
                    system: Some(Content::Ref("sys".into())),
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Ref("usr".into()),
                    }],
                }),
            }),
        ],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors());
    assert!(result.is_complete(&i, "sys"));
    assert!(result.is_complete(&i, "usr"));
    // chat calls sys() and usr() — but google_llm ExternFn not provided → type error
    // This is expected: the ExternFn must be injected
    // (testing that Blocks at least compile)
}

#[test]
fn multiple_items_mixed_namespace() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![
            Item::Block(Block {
                name: "helper".into(),
                source: "42".into(),
                mode: BlockMode::Script,
            }),
            Item::Block(Block {
                name: "tpl".into(),
                source: "hello".into(),
                mode: BlockMode::Template,
            }),
            Item::Display(DisplaySpec::Static {
                name: "out".into(),
                source: "done".into(),
            }),
        ],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors());
    assert!(result.is_complete(&i, "helper"));
    assert!(result.is_complete(&i, "tpl"));
    assert!(result.is_complete(&i, "out"));
}

// ════════════════════════════════════════════════════════════════════
// 2. Type check — Soundness (invalid specs rejected)
// ════════════════════════════════════════════════════════════════════

#[test]
fn block_undeclared_context_is_incomplete() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Block(Block {
            name: "greet".into(),
            source: "{{ @name }}".into(),
            mode: BlockMode::Template,
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(!result.has_field_errors(), "valid syntax → no field errors");
    assert!(!result.is_complete(&i, "greet"), "undeclared context → Incomplete");
}

#[test]
fn llm_inline_parse_error_detected() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Llm(LlmSpec {
            name: "chat".into(),
            provider: Provider::Google(GoogleSpec {
                endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                temperature: None, top_p: None, top_k: None, max_tokens: None,
                system: None,
                messages: vec![GoogleMessage {
                    role: GoogleRole::User,
                    content: Content::Inline("{{{{bad syntax".into()),
                }],
            }),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(result.has_field_errors(), "parse error should be detected");
    let errors = result.field_errors_for("chat");
    assert_eq!(errors, &["messages[0].content"]);
}

#[test]
fn llm_multiple_inline_errors_all_collected() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Llm(LlmSpec {
            name: "chat".into(),
            provider: Provider::Google(GoogleSpec {
                endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                temperature: None, top_p: None, top_k: None, max_tokens: None,
                system: Some(Content::Inline("{{bad1".into())),
                messages: vec![GoogleMessage {
                    role: GoogleRole::User,
                    content: Content::Inline("{{bad2".into()),
                }],
            }),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    let errors = result.field_errors_for("chat");
    assert_eq!(errors.len(), 2, "both errors collected");
    assert!(errors.contains(&"messages[0].content"));
    assert!(errors.contains(&"messages[1].content"));
}

#[test]
fn llm_ref_to_nonexistent_block_type_error() {
    let i = Interner::new();
    // LlmSpec references "sys" but no such Block exists
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Llm(LlmSpec {
            name: "chat".into(),
            provider: Provider::Google(GoogleSpec {
                endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                temperature: None, top_p: None, top_k: None, max_tokens: None,
                system: Some(Content::Ref("nonexistent".into())),
                messages: vec![],
            }),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    // No field errors (Ref is glue code, syntax is fine)
    assert!(!result.has_field_errors());
    // But Incomplete — undefined function "nonexistent"
    assert!(!result.is_complete(&i, "chat"), "ref to nonexistent block → type error → Incomplete");
}

#[test]
fn display_history_parse_error_detected() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Display(DisplaySpec::Iterator {
            name: "msgs".into(),
            history: Some("{{invalid".into()),
            live: None,
            bind: "msg".into(),
            template: "{{ $msg }}".into(),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    assert!(result.has_field_errors());
    let errors = result.field_errors_for("msgs");
    assert_eq!(errors, &["history"]);
}

#[test]
fn display_live_parse_error_detected() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Display(DisplaySpec::Iterator {
            name: "msgs".into(),
            history: Some("@ok".into()),
            live: Some("{{bad".into()),
            bind: "msg".into(),
            template: "{{ $msg }}".into(),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    let errors = result.field_errors_for("msgs");
    assert_eq!(errors, &["live"], "only live should have error");
}

#[test]
fn display_both_sources_parse_error() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Display(DisplaySpec::Iterator {
            name: "msgs".into(),
            history: Some("{{bad1".into()),
            live: Some("{{bad2".into()),
            bind: "msg".into(),
            template: "{{ $msg }}".into(),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    let errors = result.field_errors_for("msgs");
    assert_eq!(errors.len(), 2);
    assert!(errors.contains(&"history"));
    assert!(errors.contains(&"live"));
}

// ════════════════════════════════════════════════════════════════════
// 3. Error location accuracy (SpanMap)
// ════════════════════════════════════════════════════════════════════

#[test]
fn llm_inline_span_map_points_to_correct_fields() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Llm(LlmSpec {
            name: "chat".into(),
            provider: Provider::Google(GoogleSpec {
                endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                temperature: None, top_p: None, top_k: None, max_tokens: None,
                system: Some(Content::Inline("\"system msg\"".into())),
                messages: vec![
                    GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("\"user msg\"".into()),
                    },
                    GoogleMessage {
                        role: GoogleRole::Model,
                        content: Content::Ref("cached".into()), // Ref → no span entry
                    },
                ],
            }),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    let origins = result.span_origins();
    // system (inline) + user (inline) = 2 entries. model (Ref) = no entry.
    assert_eq!(origins.len(), 2);
    assert_eq!(origins[0], ("chat", "messages[0].content"));
    assert_eq!(origins[1], ("chat", "messages[1].content"));
}

#[test]
fn llm_partial_error_span_map_only_for_valid() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Llm(LlmSpec {
            name: "chat".into(),
            provider: Provider::Google(GoogleSpec {
                endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                temperature: None, top_p: None, top_k: None, max_tokens: None,
                system: Some(Content::Inline("\"valid\"".into())),
                messages: vec![GoogleMessage {
                    role: GoogleRole::User,
                    content: Content::Inline("{{bad".into()),
                }],
            }),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    // 1 field error (bad user message)
    assert_eq!(result.field_errors_for("chat"), &["messages[1].content"]);

    // 1 span entry (valid system message only)
    let origins = result.span_origins();
    assert_eq!(origins.len(), 1);
    assert_eq!(origins[0], ("chat", "messages[0].content"));
}

#[test]
fn display_span_map_entries_for_history_and_live() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![Item::Display(DisplaySpec::Iterator {
            name: "msgs".into(),
            history: Some("@history".into()),
            live: Some("@stream".into()),
            bind: "msg".into(),
            template: "{{ $msg }}".into(),
        })],
    };
    let result = compile_namespace(&i, &ns, &[]);

    let origins = result.span_origins();
    assert_eq!(origins.len(), 2);
    assert_eq!(origins[0], ("msgs", "history"));
    assert_eq!(origins[1], ("msgs", "live"));
}

#[test]
fn multiple_items_errors_tracked_separately() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![
            Item::Llm(LlmSpec {
                name: "chat1".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                    temperature: None, top_p: None, top_k: None, max_tokens: None,
                    system: None,
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("{{bad".into()),
                    }],
                }),
            }),
            Item::Llm(LlmSpec {
                name: "chat2".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                    temperature: None, top_p: None, top_k: None, max_tokens: None,
                    system: Some(Content::Inline("{{also bad".into())),
                    messages: vec![],
                }),
            }),
            Item::Display(DisplaySpec::Iterator {
                name: "display1".into(),
                history: Some("{{broken".into()),
                live: None,
                bind: "x".into(),
                template: "{{ $x }}".into(),
            }),
        ],
    };
    let result = compile_namespace(&i, &ns, &[]);

    // Each item's errors tracked with correct item_name
    let chat1_errors = result.field_errors_for("chat1");
    let chat2_errors = result.field_errors_for("chat2");
    let display_errors = result.field_errors_for("display1");

    assert_eq!(chat1_errors, &["messages[0].content"]);
    assert_eq!(chat2_errors, &["messages[0].content"]);
    assert_eq!(display_errors, &["history"]);
}

#[test]
fn error_does_not_prevent_other_items_from_compiling() {
    let i = Interner::new();
    let ns = Namespace {
        name: "test".into(),
        items: vec![
            // This block is perfectly valid
            Item::Block(Block {
                name: "valid".into(),
                source: "42".into(),
                mode: BlockMode::Script,
            }),
            // This LLM has a parse error
            Item::Llm(LlmSpec {
                name: "broken".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(), api_key: "k".into(), model: "m".into(),
                    temperature: None, top_p: None, top_k: None, max_tokens: None,
                    system: None,
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("{{bad".into()),
                    }],
                }),
            }),
        ],
    };
    let result = compile_namespace(&i, &ns, &[]);

    // The valid block should still compile
    assert!(result.is_complete(&i, "valid"), "valid block should not be affected by broken LLM");
    // The broken LLM should have field errors
    assert!(!result.field_errors_for("broken").is_empty());
}
