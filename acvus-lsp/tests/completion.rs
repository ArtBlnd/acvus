//! Completion through the session's public API: what is offered is what
//! the checker would resolve where the cursor is.

use acvus_extern::{Externs, TypesOnly};
use acvus_lsp::{
    CompletionItem, CompletionKind, DocId, Document, Hover, LspError, LspSession, Mode,
};
use acvus_mir::graph::{Bindings, CompilationGraph, Context, Function, QualifiedRef};
use acvus_mir::ty::{
    Effect, ParamTerm, PolyBuilder, PolyParam, Ty, TyTerm, TypeRegistry, lift_to_poly,
};
use acvus_mir::typeck::BodyView;
use acvus_utils::{Freeze, Interner};

fn root_contexts(interner: &Interner, ctx: &[(&str, Ty)]) -> Vec<Context> {
    ctx.iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_to_poly(ty),
        })
        .collect()
}

fn environment(
    contexts: Vec<Context>,
    functions: Vec<Function>,
    types: TypeRegistry,
) -> CompilationGraph {
    CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(types),
        bindings: Bindings::default(),
        entry: None,
    }
}

fn bare(contexts: Vec<Context>) -> CompilationGraph {
    environment(contexts, vec![], TypeRegistry::default())
}

fn with_std(interner: &Interner, contexts: Vec<Context>) -> CompilationGraph {
    let Externs {
        functions, types, ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), interner)
        .expect("standard registries combine");
    environment(contexts, functions, types)
}

fn document(interner: &Interner, name: &str, mode: Mode, params: Vec<PolyParam>) -> Document {
    let mut pb = PolyBuilder::new();
    Document {
        qref: QualifiedRef::root(interner.intern(name)),
        mode,
        ty: TyTerm::Fn {
            params,
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
        },
    }
}

fn open(
    interner: &Interner,
    environment: CompilationGraph,
    mode: Mode,
    source: &str,
) -> (LspSession, DocId) {
    let mut session = LspSession::new(interner, environment);
    let id = session.open(document(interner, "test", mode, vec![]), source);
    (session, id)
}

/// The offset of the `nth` occurrence of `needle` in `source`.
fn nth(source: &str, needle: &str, n: usize) -> usize {
    source
        .match_indices(needle)
        .nth(n)
        .map(|(at, _)| at)
        .unwrap_or_else(|| panic!("`{needle}` occurs {} times in {source:?}", n + 1))
}

fn of_kind(items: &[CompletionItem], kind: CompletionKind) -> Vec<&CompletionItem> {
    items.iter().filter(|item| item.kind == kind).collect()
}

fn labels(items: &[&CompletionItem]) -> Vec<String> {
    items.iter().map(|item| item.label.clone()).collect()
}

fn hovered(session: &LspSession, doc: DocId, offset: usize) -> String {
    session
        .hover(doc, offset)
        .unwrap_or_else(|| panic!("nothing recorded at {offset}"))
        .ty
}

#[test]
fn a_local_is_offered_at_the_type_the_checker_binds() {
    let i = Interner::new();
    let source = "let alpha = 1; let beta = \"b\"; al";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let items = session.completions(doc, source.len());
    let locals = of_kind(&items, CompletionKind::Local);
    assert_eq!(labels(&locals), ["alpha"]);
    assert_eq!(
        locals[0].detail,
        hovered(&session, doc, nth(source, "alpha", 0))
    );
    assert_eq!(locals[0].insert_text, "alpha");
    assert!(items.iter().all(|item| item.label != "beta"));
}

#[test]
fn a_name_bound_in_a_block_is_not_offered_after_it() {
    let i = Interner::new();
    let source = "let y = { let hidden = 2; hidden }; hid";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let items = session.completions(doc, source.len());
    assert!(
        of_kind(&items, CompletionKind::Local).is_empty(),
        "{items:?}"
    );
    let inside = nth(source, "hidden }", 0) + "hid".len();
    let items = session.completions(doc, inside);
    assert_eq!(labels(&of_kind(&items, CompletionKind::Local)), ["hidden"]);
}

#[test]
fn a_lambda_parameter_is_offered_only_in_its_body() {
    let i = Interner::new();
    let source = "let f = |param| -> pa; pa";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let in_body = nth(source, "pa;", 0) + "pa".len();
    let items = session.completions(doc, in_body);
    assert_eq!(labels(&of_kind(&items, CompletionKind::Local)), ["param"]);
    let items = session.completions(doc, source.len());
    assert!(
        of_kind(&items, CompletionKind::Local).is_empty(),
        "{items:?}"
    );
}

#[test]
fn a_shadowed_name_is_offered_once_at_the_inner_binding() {
    let i = Interner::new();
    let source = "let v = 1; let v = \"s\"; v";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let items = session.completions(doc, source.len());
    let locals = of_kind(&items, CompletionKind::Local);
    assert_eq!(labels(&locals), ["v"]);
    assert_eq!(
        locals[0].detail,
        hovered(&session, doc, nth(source, "v", 1))
    );
    assert_ne!(
        locals[0].detail,
        hovered(&session, doc, nth(source, "v", 0))
    );
}

#[test]
fn a_context_is_offered_after_at_by_its_prefix() {
    let i = Interner::new();
    let contexts = root_contexts(&i, &[("name", Ty::String), ("count", Ty::I64)]);
    let source = "@na";
    let (session, doc) = open(&i, bare(contexts), Mode::Script, source);
    let items = session.completions(doc, source.len());
    assert_eq!(
        items,
        [CompletionItem {
            label: "@name".to_string(),
            kind: CompletionKind::Context,
            detail: Ty::String.display(&i).to_string(),
            insert_text: "name".to_string(),
        }]
    );
}

#[test]
fn the_declared_inputs_are_offered_after_dollar() {
    let i = Interner::new();
    let params = vec![
        ParamTerm::new(i.intern("who"), lift_to_poly(&Ty::String)),
        ParamTerm::new(i.intern("n"), lift_to_poly(&Ty::I64)),
    ];
    let mut session = LspSession::new(&i, bare(vec![]));
    let source = "let local = 1; $";
    let doc = session.open(document(&i, "test", Mode::Script, params), source);
    let items = session.completions(doc, source.len());
    assert_eq!(
        items,
        [
            CompletionItem {
                label: "$n".to_string(),
                kind: CompletionKind::Param,
                detail: Ty::I64.display(&i).to_string(),
                insert_text: "n".to_string(),
            },
            CompletionItem {
                label: "$who".to_string(),
                kind: CompletionKind::Param,
                detail: Ty::String.display(&i).to_string(),
                insert_text: "who".to_string(),
            },
        ]
    );
}

/// The typed document parses, so hover shows the types the checker
/// recorded; the document as it is while `o.` is typed does not, and is
/// offered the same.
#[test]
fn an_object_offers_its_fields_at_their_types() {
    let i = Interner::new();
    let source = "let o = { alpha: 1, beta: true, }; o.zz";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let after_dot = nth(source, "o.", 0) + "o.".len();
    let items = session.completions(doc, after_dot);
    let fields = of_kind(&items, CompletionKind::Field);
    assert_eq!(labels(&fields), ["alpha", "beta"]);
    assert_eq!(
        fields[0].detail,
        hovered(&session, doc, nth(source, "1", 0))
    );
    assert_eq!(
        fields[1].detail,
        hovered(&session, doc, nth(source, "true", 0))
    );

    let typing = &source[..after_dot];
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, typing);
    assert!(!session.diagnostics(doc).is_empty(), "`o.` does not parse");
    assert_eq!(session.completions(doc, typing.len()), items);
}

/// `string::trim` takes `&str`, which a `&str` receiver reaches by a
/// reborrow; `powi` takes an `f64` by value, which no string is.
#[test]
fn a_string_receiver_is_offered_the_functions_that_take_it() {
    let i = Interner::new();
    let source = "let s = \"b\"; s.";
    let (session, doc) = open(&i, with_std(&i, vec![]), Mode::Script, source);
    let items = session.completions(doc, source.len());
    let methods = labels(&of_kind(&items, CompletionKind::Method));
    assert!(methods.contains(&"trim".to_string()), "{methods:?}");
    assert!(!methods.contains(&"powi".to_string()), "{methods:?}");
    assert!(of_kind(&items, CompletionKind::Function).is_empty());

    let named = "po";
    let (session, doc) = open(&i, with_std(&i, vec![]), Mode::Script, named);
    let functions = labels(&of_kind(
        &session.completions(doc, named.len()),
        CompletionKind::Function,
    ));
    assert!(functions.contains(&"powi".to_string()), "{functions:?}");
}

/// A `String` place is lent, and `trim`'s `&str` is a view of it.
#[test]
fn a_string_context_receiver_reaches_a_view_taking_function() {
    let i = Interner::new();
    let contexts = root_contexts(&i, &[("name", Ty::String)]);
    let source = "let s = @name; s.tr";
    let (session, doc) = open(&i, with_std(&i, contexts), Mode::Script, source);
    let items = session.completions(doc, source.len());
    let methods = labels(&of_kind(&items, CompletionKind::Method));
    assert!(methods.contains(&"trim".to_string()), "{methods:?}");
    assert!(
        methods.iter().all(|name| name.starts_with("tr")),
        "{methods:?}"
    );
}

/// `sort` is the one function of its name, and its first parameter is a
/// `&mut [T]` whose instances are over scalars and text: an array of
/// options has the head it lends, and the checker still refuses the call,
/// for no instance takes that element.
#[test]
fn a_single_candidate_is_offered_only_where_the_checker_takes_the_receiver() {
    let i = Interner::new();
    let options = "let v = [Some(1)]; v.";
    let (session, doc) = open(&i, with_std(&i, vec![]), Mode::Script, options);
    let methods = labels(&of_kind(
        &session.completions(doc, options.len()),
        CompletionKind::Method,
    ));
    assert!(!methods.contains(&"sort".to_string()), "{methods:?}");
    assert!(methods.contains(&"len".to_string()), "{methods:?}");

    let integers = "let v = [2, 1]; v.";
    let (session, doc) = open(&i, with_std(&i, vec![]), Mode::Script, integers);
    let methods = labels(&of_kind(
        &session.completions(doc, integers.len()),
        CompletionKind::Method,
    ));
    assert!(methods.contains(&"sort".to_string()), "{methods:?}");
}

#[test]
fn a_qualifier_offers_only_its_namespace() {
    let i = Interner::new();
    let environment = with_std(&i, vec![]);
    let in_string: Vec<String> = environment
        .functions
        .iter()
        .filter(|f| f.qref.namespace.map(|ns| i.resolve(ns)) == Some("string"))
        .map(|f| i.resolve(f.qref.name).to_string())
        .collect();
    let source = "string::";
    let (session, doc) = open(&i, environment, Mode::Script, source);
    let items = session.completions(doc, source.len());
    assert!(!items.is_empty());
    assert!(
        items
            .iter()
            .all(|item| item.kind == CompletionKind::Function && in_string.contains(&item.label)),
        "{items:?}"
    );
    assert!(items.iter().any(|item| item.label == "trim"));
    assert!(items.iter().all(|item| item.label != "powi"));
}

#[test]
fn a_template_completes_inside_code_only() {
    let i = Interner::new();
    let source = "% let alpha = 1\n% let beta = al\nal text {{ al }}\n";
    let (session, doc) = open(&i, bare(vec![]), Mode::Template, source);

    let in_text = nth(source, "al text", 0) + "al".len();
    assert_eq!(session.completions(doc, in_text), []);

    let in_tag = nth(source, "al }}", 0) + "al".len();
    let items = session.completions(doc, in_tag);
    assert_eq!(labels(&of_kind(&items, CompletionKind::Local)), ["alpha"]);

    let on_line = nth(source, "al\n", 0) + "al".len();
    let items = session.completions(doc, on_line);
    assert_eq!(labels(&of_kind(&items, CompletionKind::Local)), ["alpha"]);
}

#[test]
fn a_keyword_is_offered_by_its_prefix() {
    let i = Interner::new();
    let source = "ma";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let items = session.completions(doc, source.len());
    assert_eq!(
        items,
        [CompletionItem {
            label: "match".to_string(),
            kind: CompletionKind::Keyword,
            detail: "keyword".to_string(),
            insert_text: "match".to_string(),
        }]
    );
}

#[test]
fn a_probe_that_does_not_parse_offers_what_needs_no_tree() {
    let i = Interner::new();
    let contexts = root_contexts(&i, &[("name", Ty::String)]);
    let source = "let alpha = 1; ) ";
    let (session, doc) = open(&i, with_std(&i, contexts), Mode::Script, source);
    let items = session.completions(doc, source.len());
    assert!(!of_kind(&items, CompletionKind::Keyword).is_empty());
    assert!(!of_kind(&items, CompletionKind::Function).is_empty());
    assert!(
        items.iter().all(|item| matches!(
            item.kind,
            CompletionKind::Keyword | CompletionKind::Function
        )),
        "{items:?}"
    );

    let member = "let s = \"b\"; ) s.";
    let (session, doc) = open(&i, with_std(&i, vec![]), Mode::Script, member);
    assert_eq!(session.completions(doc, member.len()), []);

    let context = "let alpha = 1; ) @";
    let contexts = root_contexts(&i, &[("name", Ty::String)]);
    let (session, doc) = open(&i, bare(contexts), Mode::Script, context);
    assert_eq!(
        labels(&of_kind(
            &session.completions(doc, context.len()),
            CompletionKind::Context
        )),
        ["@name"]
    );
}

#[test]
fn a_statement_being_written_completes_from_what_parsed() {
    let i = Interner::new();
    let source = "let o = { alpha: 1, beta: true, }; let y = o.";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    assert!(
        !session.diagnostics(doc).is_empty(),
        "the `let` is unfinished"
    );
    let items = session.completions(doc, source.len());
    assert_eq!(
        labels(&of_kind(&items, CompletionKind::Field)),
        ["alpha", "beta"]
    );

    let source = "let = 3;\nlet alpha = 1; al";
    let (session, doc) = open(&i, bare(vec![]), Mode::Script, source);
    let items = session.completions(doc, source.len());
    assert_eq!(labels(&of_kind(&items, CompletionKind::Local)), ["alpha"]);
}

#[test]
fn completion_leaves_the_graph_unchanged() {
    let i = Interner::new();
    let contexts = root_contexts(&i, &[("name", Ty::String)]);
    let mut session = LspSession::new(&i, with_std(&i, contexts));
    let helper_source = "let n = @name; n";
    let helper = session.open(document(&i, "helper", Mode::Script, vec![]), helper_source);
    let main_source = "let s = helper(); let o = { a: s, }; o.a.tr";
    let main = session.open(document(&i, "main", Mode::Script, vec![]), main_source);
    let broken_source = "let t = 1; t * \"x\"; t";
    let broken = session.open(document(&i, "broken", Mode::Script, vec![]), broken_source);
    let documents = [
        Opened {
            doc: helper,
            source: helper_source,
        },
        Opened {
            doc: main,
            source: main_source,
        },
        Opened {
            doc: broken,
            source: broken_source,
        },
    ];

    let before: Vec<Observed> = documents
        .iter()
        .map(|opened| opened.observed(&session))
        .collect();
    for opened in &documents {
        for cursor in 0..=opened.source.len() {
            session.completions(opened.doc, cursor);
        }
    }
    for (opened, before) in documents.iter().zip(before) {
        let after = opened.observed(&session);
        assert_eq!(before.diagnostics, after.diagnostics);
        assert_eq!(before.hovers, after.hovers);
        assert!(
            std::ptr::eq(&*before.view, &*after.view),
            "a body was checked again"
        );
    }
}

struct Opened<'s> {
    doc: DocId,
    source: &'s str,
}

struct Observed {
    diagnostics: Vec<LspError>,
    hovers: Vec<Option<Hover>>,
    view: Freeze<BodyView>,
}

impl Opened<'_> {
    fn observed(&self, session: &LspSession) -> Observed {
        let qref = session
            .function_ref(self.doc)
            .expect("the document is open");
        Observed {
            diagnostics: session.diagnostics(self.doc),
            hovers: (0..self.source.len())
                .map(|offset| session.hover(self.doc, offset))
                .collect(),
            view: session.graph().view(qref).expect("every body was checked"),
        }
    }
}
