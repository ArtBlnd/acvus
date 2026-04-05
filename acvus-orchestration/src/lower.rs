//! Spec → CompilationGraph lowering.
//!
//! Converts orchestration specs (Block, LlmSpec, DisplaySpec) into a
//! CompilationGraph that the acvus compiler pipeline can process.
//!
//! Lowering strategy:
//! - **User content** (inline strings, expressions): parsed via `parse_expr`.
//!   Parse errors are collected per-field. Type errors map back via SpanMap.
//! - **Glue structure** (Object literals, List, FuncCall): AST directly constructed.
//!   No parse errors possible. Type errors here = lowerer bug → panic.

use acvus_ast::{AstId, Expr, Literal, ObjectExprField, RefKind, Script, Span};
use acvus_mir::graph::{
    CompilationGraph, Constraint, FnConstraint, FnKind, Function, ParsedAst, QualifiedRef,
    Signature,
};
use acvus_mir::ty::{EffectConstraint, Ty};
use acvus_utils::{Astr, Freeze, Interner};

use crate::spec::{Block, BlockMode, Content, DisplaySpec, Item, LlmSpec, Namespace, Provider};

// ── Error types ────────────────────────────────────────────────────

/// A field-level error from lowering.
#[derive(Debug)]
pub struct FieldError {
    /// Which spec item this error belongs to.
    pub item_name: String,
    /// Which field within the item.
    pub field: String,
    /// The parse error.
    pub error: acvus_ast::ParseError,
}

// ── SpanMap ────────────────────────────────────────────────────────

/// Maps an AST Span (in a generated function) back to the spec field it came from.
#[derive(Debug, Clone)]
pub struct SpanEntry {
    /// The AST span of the user-content expression.
    pub span: Span,
    /// Which spec field produced this expression.
    pub origin: SpecOrigin,
}

/// Identifies a field in the original spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecOrigin {
    LlmField { llm_name: String, field: String },
    DisplayField { display_name: String, field: String },
}

/// Accumulated span mappings from lowering.
#[derive(Debug, Default)]
pub struct SpanMap {
    pub entries: Vec<SpanEntry>,
}

impl SpanMap {
    /// Find the spec origin for a given AST span.
    pub fn resolve(&self, span: Span) -> Option<&SpecOrigin> {
        self.entries
            .iter()
            .find(|e| e.span == span)
            .map(|e| &e.origin)
    }
}

// ── Lowerer output ─────────────────────────────────────────────────

pub struct LowerOutput {
    pub graph: CompilationGraph,
    pub span_map: SpanMap,
    /// Per-field parse errors from user content.
    pub field_errors: Vec<FieldError>,
}

/// Lower a Namespace spec into a CompilationGraph.
///
/// `extern_fns`: pre-configured ExternFn functions (e.g., config-bound LLM callables).
pub fn lower_namespace(
    interner: &Interner,
    ns: &Namespace,
    extern_fns: &[Function],
) -> LowerOutput {
    let ns_name = interner.intern(&ns.name);
    let mut functions: Vec<Function> = extern_fns.to_vec();
    let contexts = Vec::new();
    let mut span_map = SpanMap::default();
    let mut field_errors = Vec::new();

    for item in &ns.items {
        match item {
            Item::Block(block) => {
                let result = lower_block(interner, block, ns_name);
                functions.push(result.function);
                field_errors.extend(result.field_errors);
            }
            Item::Llm(llm) => {
                let result = lower_llm(interner, llm, ns_name);
                functions.push(result.function);
                span_map.entries.extend(result.span_entries);
                field_errors.extend(result.field_errors);
            }
            Item::Display(display) => {
                let result = lower_display(interner, display, ns_name);
                functions.extend(result.functions);
                span_map.entries.extend(result.span_entries);
                field_errors.extend(result.field_errors);
            }
        }
    }

    LowerOutput {
        graph: CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        },
        span_map,
        field_errors,
    }
}

// ── AST construction helpers ───────────────────────────────────────

/// Synthetic span for lowerer-generated AST nodes.
/// All glue code uses this — if a type error hits this span, it's a lowerer bug.
const GLUE_SPAN: Span = Span::ZERO;

/// Build a string literal expression.
fn str_lit(s: &str) -> Expr {
    Expr::Literal {
        id: AstId::alloc(),
        value: Literal::String(s.to_string()),
        span: GLUE_SPAN,
    }
}

/// Build an identifier expression (bare name).
fn ident(interner: &Interner, name: &str) -> Expr {
    Expr::Ident {
        id: AstId::alloc(),
        name: QualifiedRef::root(interner.intern(name)),
        ref_kind: RefKind::Value,
        span: GLUE_SPAN,
    }
}

/// Build a function call expression: `func(args...)`.
fn call(func: Expr, args: Vec<Expr>) -> Expr {
    Expr::FuncCall {
        id: AstId::alloc(),
        func: Box::new(func),
        args,
        span: GLUE_SPAN,
    }
}

/// Build a no-arg function call: `name()`.
fn call0(interner: &Interner, name: &str) -> Expr {
    call(ident(interner, name), vec![])
}

/// Build a list expression: `[a, b, c]`.
fn list(items: Vec<Expr>) -> Expr {
    Expr::List {
        id: AstId::alloc(),
        head: items,
        rest: None,
        tail: vec![],
        span: GLUE_SPAN,
    }
}

/// Build an object expression: `{ key1: val1, key2: val2 }`.
fn object(interner: &Interner, fields: Vec<(&str, Expr)>) -> Expr {
    Expr::Object {
        id: AstId::alloc(),
        fields: fields
            .into_iter()
            .map(|(k, v)| ObjectExprField {
                id: AstId::alloc(),
                key: interner.intern(k),
                value: v,
                span: GLUE_SPAN,
            })
            .collect(),
        span: GLUE_SPAN,
    }
}

/// Build a pipe expression: `left | right`.
fn pipe(left: Expr, right: Expr) -> Expr {
    Expr::Pipe {
        id: AstId::alloc(),
        left: Box::new(left),
        right: Box::new(right),
        span: GLUE_SPAN,
    }
}

/// Build a Script with no statements, just a tail expression.
fn script_tail(tail: Expr) -> Script {
    Script {
        id: AstId::alloc(),
        stmts: vec![],
        tail: Some(Box::new(tail)),
        span: GLUE_SPAN,
    }
}

/// Build a Script with statements and a tail expression.
fn script(stmts: Vec<acvus_ast::Stmt>, tail: Expr) -> Script {
    Script {
        id: AstId::alloc(),
        stmts,
        tail: Some(Box::new(tail)),
        span: GLUE_SPAN,
    }
}

/// Build a bind statement: `name = expr;`
fn bind(interner: &Interner, name: &str, expr: Expr) -> acvus_ast::Stmt {
    acvus_ast::Stmt::Bind {
        id: AstId::alloc(),
        name: interner.intern(name),
        expr,
        span: GLUE_SPAN,
    }
}

// ── Block lowering ─────────────────────────────────────────────────

struct BlockLowerResult {
    function: Function,
    field_errors: Vec<FieldError>,
}

fn lower_block(interner: &Interner, block: &Block, ns_name: Astr) -> BlockLowerResult {
    let mut field_errors = Vec::new();

    let parsed_ast = match block.mode {
        BlockMode::Script => match acvus_ast::parse_script(interner, &block.source) {
            Ok(ast) => ParsedAst::Script(ast),
            Err(err) => {
                field_errors.push(FieldError {
                    item_name: block.name.clone(),
                    field: "source".into(),
                    error: err,
                });
                ParsedAst::Script(script_tail(str_lit("")))
            }
        },
        BlockMode::Template => match acvus_ast::parse(interner, &block.source) {
            Ok(ast) => ParsedAst::Template(ast),
            Err(err) => {
                field_errors.push(FieldError {
                    item_name: block.name.clone(),
                    field: "source".into(),
                    error: err,
                });
                ParsedAst::Template(acvus_ast::Template {
                    id: AstId::alloc(),
                    body: vec![],
                    span: GLUE_SPAN,
                })
            }
        },
    };

    let output = match block.mode {
        BlockMode::Template => Constraint::Exact(Ty::String),
        BlockMode::Script => Constraint::Inferred,
    };

    let qref = QualifiedRef::qualified(ns_name, interner.intern(&block.name));
    BlockLowerResult {
        function: Function {
            qref,
            kind: FnKind::Local(parsed_ast),
            constraint: FnConstraint {
                signature: Some(Signature { params: vec![] }),
                output,
                effect: Some(EffectConstraint::read_only()),
                hint: None,
            },
        },
        field_errors,
    }
}

// ── LLM lowering ──────────────────────────────────────────────────

struct LlmLowerResult {
    function: Function,
    span_entries: Vec<SpanEntry>,
    field_errors: Vec<FieldError>,
}

/// LlmSpec → Script Function (LocalAst) that calls the pre-configured ExternFn.
///
/// Each message becomes an Object literal `{ role: "...", content: ... }`.
/// Content::Ref → function call, Content::Inline → parsed expression.
/// All messages are collected into a List and passed to the ExternFn.
fn lower_llm(interner: &Interner, llm: &LlmSpec, ns_name: Astr) -> LlmLowerResult {
    let (extern_fn_name, messages) = match &llm.provider {
        Provider::Google(spec) => ("google_llm", build_google_messages(spec)),
        Provider::OpenAI(_) => todo!("OpenAI lowering"),
        Provider::Anthropic(_) => todo!("Anthropic lowering"),
    };

    let mut stmts = Vec::new();
    let mut stub_exprs = Vec::new();
    let mut span_entries = Vec::new();
    let mut field_errors = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        let content_expr = match &msg.content {
            MessageContent::Ref(block_name) => {
                // block_name() — glue code, no parse error possible
                call0(interner, block_name)
            }
            MessageContent::Inline(text) => {
                // Parse user content — errors collected per-field
                match acvus_ast::parse_expr(interner, text) {
                    Ok(expr) => {
                        // Record span for type error mapping
                        span_entries.push(SpanEntry {
                            span: expr.span(),
                            origin: SpecOrigin::LlmField {
                                llm_name: llm.name.clone(),
                                field: format!("messages[{i}].content"),
                            },
                        });
                        expr
                    }
                    Err(err) => {
                        field_errors.push(FieldError {
                            item_name: llm.name.clone(),
                            field: format!("messages[{i}].content"),
                            error: err,
                        });
                        // Poison: use empty string so lowering can continue
                        str_lit("")
                    }
                }
            }
            MessageContent::Iterator(_) => {
                todo!("Iterator content lowering")
            }
        };

        // { role: "...", content: <expr> }
        let msg_obj = object(
            interner,
            vec![("role", str_lit(&msg.role)), ("content", content_expr)],
        );

        let stub_name = format!("__s{i}");
        stmts.push(bind(interner, &stub_name, msg_obj));
        stub_exprs.push(ident(interner, &stub_name));
    }

    // Tail: extern_fn([__s0, __s1, ...])
    let messages_list = list(stub_exprs);
    let tail = call(ident(interner, extern_fn_name), vec![messages_list]);
    let ast = script(stmts, tail);

    let qref = QualifiedRef::qualified(ns_name, interner.intern(&llm.name));
    let function = Function {
        qref,
        kind: FnKind::Local(ParsedAst::Script(ast)),
        constraint: FnConstraint {
            signature: Some(Signature { params: vec![] }),
            output: Constraint::Inferred,
            effect: None, // LLM calls involve IO — no constraint
            hint: None,
        },
    };

    LlmLowerResult {
        function,
        span_entries,
        field_errors,
    }
}

struct LowerMessage {
    role: String,
    content: MessageContent,
}

enum MessageContent {
    Ref(String),
    Inline(String),
    Iterator(String),
}

fn build_google_messages(spec: &crate::spec::GoogleSpec) -> Vec<LowerMessage> {
    let mut messages = Vec::new();

    if let Some(system) = &spec.system {
        messages.push(LowerMessage {
            role: "system".into(),
            content: content_to_lower(system),
        });
    }

    for msg in &spec.messages {
        let role = match msg.role {
            crate::spec::GoogleRole::User => "user",
            crate::spec::GoogleRole::Model => "model",
        };
        messages.push(LowerMessage {
            role: role.into(),
            content: content_to_lower(&msg.content),
        });
    }

    messages
}

fn content_to_lower(content: &Content) -> MessageContent {
    match content {
        Content::Inline(s) => MessageContent::Inline(s.clone()),
        Content::Ref(name) => MessageContent::Ref(name.clone()),
        Content::Iterator(expr) => MessageContent::Iterator(expr.clone()),
    }
}

// ── Display lowering ──────────────────────────────────────────────

struct DisplayLowerResult {
    functions: Vec<Function>,
    span_entries: Vec<SpanEntry>,
    field_errors: Vec<FieldError>,
}

/// DisplaySpec → Function(s).
///
/// - Static: 1 Template function (source as-is).
/// - Iterator: up to 3 functions:
///   - template function: `tpl(bind) -> String`
///   - history function (if Some): `@source | map(tpl)` → Iterator<String>
///   - live function (if Some): `@source | map(tpl)` → Iterator<String>
fn lower_display(interner: &Interner, display: &DisplaySpec, ns_name: Astr) -> DisplayLowerResult {
    match display {
        DisplaySpec::Static { name, source } => {
            let qref = QualifiedRef::qualified(ns_name, interner.intern(name));
            let mut field_errors = Vec::new();

            let parsed = match acvus_ast::parse(interner, source) {
                Ok(ast) => ParsedAst::Template(ast),
                Err(err) => {
                    field_errors.push(FieldError {
                        item_name: name.clone(),
                        field: "source".into(),
                        error: err,
                    });
                    ParsedAst::Template(acvus_ast::Template {
                        id: AstId::alloc(),
                        body: vec![],
                        span: GLUE_SPAN,
                    })
                }
            };

            let func = Function {
                qref,
                kind: FnKind::Local(parsed),
                constraint: FnConstraint {
                    signature: Some(Signature { params: vec![] }),
                    output: Constraint::Exact(Ty::String),
                    effect: Some(EffectConstraint::read_only()),
                    hint: None,
                },
            };
            DisplayLowerResult {
                functions: vec![func],
                span_entries: vec![],
                field_errors,
            }
        }
        DisplaySpec::Iterator {
            name,
            history,
            live,
            bind: _,
            template,
        } => {
            let mut functions = Vec::new();
            let mut span_entries = Vec::new();
            let mut field_errors = Vec::new();

            // 1. Template function: tpl($bind) -> String
            //    The bind param is an ExternParam ($name) in the template.
            //    Infer discovers it in analysis_mode and infers its type from usage.
            let tpl_name = format!("__{name}_tpl");
            let tpl_qref = QualifiedRef::qualified(ns_name, interner.intern(&tpl_name));
            let tpl_parsed = match acvus_ast::parse(interner, template) {
                Ok(ast) => ParsedAst::Template(ast),
                Err(err) => {
                    field_errors.push(FieldError {
                        item_name: name.clone(),
                        field: "template".into(),
                        error: err,
                    });
                    ParsedAst::Template(acvus_ast::Template {
                        id: AstId::alloc(),
                        body: vec![],
                        span: GLUE_SPAN,
                    })
                }
            };

            let tpl_func = Function {
                qref: tpl_qref,
                kind: FnKind::Local(tpl_parsed),
                constraint: FnConstraint {
                    signature: None, // param discovered by Infer via $bind
                    output: Constraint::Exact(Ty::String),
                    effect: Some(EffectConstraint::read_only()),
                    hint: None,
                },
            };
            functions.push(tpl_func);

            // 2. History function (if present): @source | map(tpl)
            if let Some(history_source) = history {
                let history_name = format!("{name}_history");

                // Parse the history source expression (user content)
                match acvus_ast::parse_expr(interner, history_source) {
                    Ok(source_expr) => {
                        span_entries.push(SpanEntry {
                            span: source_expr.span(),
                            origin: SpecOrigin::DisplayField {
                                display_name: name.clone(),
                                field: "history".into(),
                            },
                        });

                        // source_expr | map(__tpl)
                        let map_call = pipe(
                            source_expr,
                            call(ident(interner, "map"), vec![ident(interner, &tpl_name)]),
                        );
                        let ast = script_tail(map_call);

                        let history_qref =
                            QualifiedRef::qualified(ns_name, interner.intern(&history_name));
                        functions.push(Function {
                            qref: history_qref,
                            kind: FnKind::Local(ParsedAst::Script(ast)),
                            constraint: FnConstraint {
                                signature: Some(Signature { params: vec![] }),
                                output: Constraint::Inferred,
                                effect: Some(EffectConstraint::read_only()),
                                hint: None,
                            },
                        });
                    }
                    Err(err) => {
                        field_errors.push(FieldError {
                            item_name: name.clone(),
                            field: "history".into(),
                            error: err,
                        });
                    }
                }
            }

            // 3. Live function (if present): @source | map(tpl)
            if let Some(live_source) = live {
                let live_name = format!("{name}_live");

                match acvus_ast::parse_expr(interner, live_source) {
                    Ok(source_expr) => {
                        span_entries.push(SpanEntry {
                            span: source_expr.span(),
                            origin: SpecOrigin::DisplayField {
                                display_name: name.clone(),
                                field: "live".into(),
                            },
                        });

                        let map_call = pipe(
                            source_expr,
                            call(ident(interner, "map"), vec![ident(interner, &tpl_name)]),
                        );
                        let ast = script_tail(map_call);

                        let live_qref =
                            QualifiedRef::qualified(ns_name, interner.intern(&live_name));
                        functions.push(Function {
                            qref: live_qref,
                            kind: FnKind::Local(ParsedAst::Script(ast)),
                            constraint: FnConstraint {
                                signature: Some(Signature { params: vec![] }),
                                output: Constraint::Inferred,
                                effect: Some(EffectConstraint::read_only()),
                                hint: None,
                            },
                        });
                    }
                    Err(err) => {
                        field_errors.push(FieldError {
                            item_name: name.clone(),
                            field: "live".into(),
                            error: err,
                        });
                    }
                }
            }

            DisplayLowerResult {
                functions,
                span_entries,
                field_errors,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::*;

    // ── Block tests ────────────────────────────────────────────────

    #[test]
    fn lower_template_block() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Block(Block {
                name: "greeting".into(),
                source: "Hello, {{ @name }}!".into(),
                mode: BlockMode::Template,
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        let func = &output.graph.functions[0];
        assert_eq!(i.resolve(func.qref.name), "greeting");
        assert!(matches!(
            func.constraint.output,
            Constraint::Exact(Ty::String)
        ));
        assert!(func.constraint.effect.is_some(), "should have effect constraint");
    }

    #[test]
    fn lower_script_block() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Block(Block {
                name: "compute".into(),
                source: "@x + 1".into(),
                mode: BlockMode::Script,
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        let func = &output.graph.functions[0];
        assert!(matches!(func.constraint.output, Constraint::Inferred));
    }

    // ── LLM tests ─────────────────────────────────────────────────

    #[test]
    fn lower_llm_ref_messages_produces_local_ast() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![
                Item::Block(Block {
                    name: "sys".into(),
                    source: "You are helpful.".into(),
                    mode: BlockMode::Template,
                }),
                Item::Llm(LlmSpec {
                    name: "chat".into(),
                    provider: Provider::Google(GoogleSpec {
                        endpoint: "https://api.google.com".into(),
                        api_key: "key".into(),
                        model: "gemini-2.0-flash".into(),
                        temperature: None,
                        top_p: None,
                        top_k: None,
                        max_tokens: None,
                        system: Some(Content::Ref("sys".into())),
                        messages: vec![GoogleMessage {
                            role: GoogleRole::User,
                            content: Content::Ref("usr".into()),
                        }],
                    }),
                }),
            ],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // Block + LLM function
        assert_eq!(output.graph.functions.len(), 2);
        let llm_func = &output.graph.functions[1];
        assert_eq!(i.resolve(llm_func.qref.name), "chat");
        assert!(matches!(llm_func.kind, FnKind::Local(ParsedAst::Script(_))));

        // Ref-only messages → no span entries, no field errors
        assert!(output.span_map.entries.is_empty());
        assert!(output.field_errors.is_empty());
    }

    #[test]
    fn lower_llm_inline_content_has_span_entries() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "https://api.google.com".into(),
                    api_key: "key".into(),
                    model: "gemini-2.0-flash".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
                    system: Some(Content::Inline("\"You are helpful.\"".into())),
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("\"Hello!\"".into()),
                    }],
                }),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // Inline content → span entries for type error mapping
        assert_eq!(output.span_map.entries.len(), 2);
        assert!(matches!(&output.span_map.entries[0].origin,
            SpecOrigin::LlmField { field, .. } if field == "messages[0].content"));
        assert!(matches!(&output.span_map.entries[1].origin,
            SpecOrigin::LlmField { field, .. } if field == "messages[1].content"));
    }

    #[test]
    fn lower_llm_inline_parse_error_collected() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "https://api.google.com".into(),
                    api_key: "key".into(),
                    model: "gemini-2.0-flash".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
                    system: None,
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("{{{{invalid".into()),
                    }],
                }),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // Parse error collected, not panicked
        assert_eq!(output.field_errors.len(), 1);
        assert_eq!(output.field_errors[0].field, "messages[0].content");
    }

    // ── Display tests ─────────────────────────────────────────────

    #[test]
    fn lower_display_static() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Static {
                name: "output".into(),
                source: "Hello!".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(output.graph.functions.len(), 1);
        let func = &output.graph.functions[0];
        assert_eq!(i.resolve(func.qref.name), "output");
        assert!(matches!(
            func.constraint.output,
            Constraint::Exact(Ty::String)
        ));
    }

    #[test]
    fn lower_display_iterator_both() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "messages".into(),
                history: Some("@history".into()),
                live: Some("@stream".into()),
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // template + history + live = 3 functions
        assert_eq!(output.graph.functions.len(), 3);
        assert_eq!(
            i.resolve(output.graph.functions[0].qref.name),
            "__messages_tpl"
        );
        assert_eq!(
            i.resolve(output.graph.functions[1].qref.name),
            "messages_history"
        );
        assert_eq!(
            i.resolve(output.graph.functions[2].qref.name),
            "messages_live"
        );

        // history and live are LocalAst
        assert!(matches!(
            output.graph.functions[1].kind,
            FnKind::Local(ParsedAst::Script(_))
        ));
        assert!(matches!(
            output.graph.functions[2].kind,
            FnKind::Local(ParsedAst::Script(_))
        ));
    }

    #[test]
    fn lower_display_iterator_history_only() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "msgs".into(),
                history: Some("@history".into()),
                live: None,
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // template + history = 2
        assert_eq!(output.graph.functions.len(), 2);
    }

    #[test]
    fn lower_display_iterator_none() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "msgs".into(),
                history: None,
                live: None,
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // template only = 1 (and it'll fail typechecking — but that's valid)
        assert_eq!(output.graph.functions.len(), 1);
    }

    // ── Extern fn injection ───────────────────────────────────────

    #[test]
    fn extern_fns_included_in_graph() {
        let i = Interner::new();
        let extern_fn = Function {
            qref: QualifiedRef::root(i.intern("google_llm")),
            kind: FnKind::Extern,
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
                hint: None,
            },
        };
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Block(Block {
                name: "a".into(),
                source: "1".into(),
                mode: BlockMode::Script,
            })],
        };
        let output = lower_namespace(&i, &ns, &[extern_fn]);

        assert_eq!(output.graph.functions.len(), 2);
        assert_eq!(i.resolve(output.graph.functions[0].qref.name), "google_llm");
    }

    // ════════════════════════════════════════════════════════════════
    // Soundness: errors detected and collected per-field
    // ════════════════════════════════════════════════════════════════

    // -- LLM parse errors --

    /// Multiple inline parse errors → all collected, none swallowed.
    #[test]
    fn llm_multiple_parse_errors_all_collected() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "https://api.google.com".into(),
                    api_key: "key".into(),
                    model: "gemini-2.0-flash".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
                    system: Some(Content::Inline("{{bad".into())),
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("also {{bad".into()),
                    }],
                }),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(
            output.field_errors.len(),
            2,
            "both errors should be collected"
        );
        assert_eq!(output.field_errors[0].field, "messages[0].content");
        assert_eq!(output.field_errors[1].field, "messages[1].content");
    }

    /// Parse error in one message, valid in another → error collected, valid proceeds.
    #[test]
    fn llm_partial_error_valid_continues() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "https://api.google.com".into(),
                    api_key: "key".into(),
                    model: "gemini-2.0-flash".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
                    system: Some(Content::Inline("\"valid system\"".into())),
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("{{bad".into()),
                    }],
                }),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // One error for the bad message
        assert_eq!(output.field_errors.len(), 1);
        assert_eq!(output.field_errors[0].field, "messages[1].content");

        // Valid system message → span entry exists
        assert_eq!(output.span_map.entries.len(), 1);
        assert_eq!(
            output.span_map.entries[0].origin,
            SpecOrigin::LlmField {
                llm_name: "chat".into(),
                field: "messages[0].content".into(),
            }
        );

        // Function still produced (with poison for bad content)
        assert_eq!(output.graph.functions.len(), 1);
        assert!(matches!(
            output.graph.functions[0].kind,
            FnKind::Local(ParsedAst::Script(_))
        ));
    }

    /// field_errors carry correct item_name.
    #[test]
    fn llm_error_has_correct_item_name() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Llm(LlmSpec {
                name: "my_chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(),
                    api_key: "k".into(),
                    model: "m".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
                    system: None,
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("{{bad".into()),
                    }],
                }),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(output.field_errors[0].item_name, "my_chat");
    }

    // -- Display parse errors --

    /// Invalid history source expression → field error.
    #[test]
    fn display_history_parse_error() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "msgs".into(),
                history: Some("{{bad expr".into()),
                live: None,
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(output.field_errors.len(), 1);
        assert_eq!(output.field_errors[0].item_name, "msgs");
        assert_eq!(output.field_errors[0].field, "history");

        // Template function still produced, but no history function
        assert_eq!(output.graph.functions.len(), 1); // only template
    }

    /// Invalid live source expression → field error.
    #[test]
    fn display_live_parse_error() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "msgs".into(),
                history: Some("@history".into()),
                live: Some("{{bad".into()),
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(output.field_errors.len(), 1);
        assert_eq!(output.field_errors[0].field, "live");

        // Template + history produced, no live
        assert_eq!(output.graph.functions.len(), 2);
    }

    /// Both history and live invalid → both errors collected.
    #[test]
    fn display_both_parse_errors() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "msgs".into(),
                history: Some("{{bad1".into()),
                live: Some("{{bad2".into()),
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(output.field_errors.len(), 2);
        assert_eq!(output.field_errors[0].field, "history");
        assert_eq!(output.field_errors[1].field, "live");

        // Only template function produced
        assert_eq!(output.graph.functions.len(), 1);
    }

    // -- SpanMap correctness --

    /// Each inline content's span maps to the correct field.
    #[test]
    fn span_map_entries_match_fields() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(),
                    api_key: "k".into(),
                    model: "m".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
                    system: Some(Content::Inline("\"sys\"".into())),
                    messages: vec![
                        GoogleMessage {
                            role: GoogleRole::User,
                            content: Content::Inline("\"usr1\"".into()),
                        },
                        GoogleMessage {
                            role: GoogleRole::Model,
                            content: Content::Ref("cached_reply".into()),
                        },
                        GoogleMessage {
                            role: GoogleRole::User,
                            content: Content::Inline("\"usr2\"".into()),
                        },
                    ],
                }),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        // 3 inline contents (system + usr1 + usr2), 1 ref (no entry)
        assert_eq!(output.span_map.entries.len(), 3);

        let fields: Vec<&str> = output
            .span_map
            .entries
            .iter()
            .map(|e| match &e.origin {
                SpecOrigin::LlmField { field, .. } => field.as_str(),
                _ => panic!("expected LlmField"),
            })
            .collect();
        assert_eq!(
            fields,
            &[
                "messages[0].content",
                "messages[1].content",
                "messages[3].content"
            ]
        );
    }

    /// Display Iterator: valid source expressions have span entries.
    #[test]
    fn display_span_map_entries() {
        let i = Interner::new();
        let ns = Namespace {
            name: "test".into(),
            defaults: vec![],
            items: vec![Item::Display(DisplaySpec::Iterator {
                name: "msgs".into(),
                history: Some("@history".into()),
                live: Some("@stream".into()),
                bind: "msg".into(),
                template: "{{ $msg }}".into(),
            })],
        };
        let output = lower_namespace(&i, &ns, &[]);

        assert_eq!(output.span_map.entries.len(), 2);

        let fields: Vec<&str> = output
            .span_map
            .entries
            .iter()
            .map(|e| match &e.origin {
                SpecOrigin::DisplayField { field, .. } => field.as_str(),
                _ => panic!("expected DisplayField"),
            })
            .collect();
        assert_eq!(fields, &["history", "live"]);
    }
}
