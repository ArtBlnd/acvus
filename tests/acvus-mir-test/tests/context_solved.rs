//! Contexts at an open type, solved across the bodies of one graph
//! (RFC-0090 rule 1), and the fetches each body keeps (RFC-0025 rule 2).

use std::collections::BTreeMap;

use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::*;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::{PolyBuilder, TyTerm};
use acvus_utils::{Freeze, Interner};

struct Body {
    name: &'static str,
    source: &'static str,
}

struct Solved {
    i: Interner,
    lowered: lower::LowerResult,
    types_by_context: BTreeMap<String, String>,
}

impl Solved {
    fn ty(&self, context: &str) -> &str {
        self.types_by_context
            .get(context)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("`@{context}` is a context of the graph"))
    }

    fn module(&self, body: &str) -> &MirModule {
        self.lowered
            .module(QualifiedRef::root(self.i.intern(body)))
            .unwrap_or_else(|| panic!("`{body}` lowers"))
    }

    fn fetches(&self, body: &str) -> Vec<String> {
        self.module(body)
            .main
            .insts
            .iter()
            .filter_map(|inst| match &inst.kind {
                InstKind::Fetch { context, .. } => Some(self.i.resolve(context.name).to_owned()),
                _ => None,
            })
            .collect()
    }

    /// Each commit of `main`, in body order: its context and whether it
    /// stores.
    fn commits(&self, body: &str) -> Vec<(String, bool)> {
        self.module(body)
            .main
            .insts
            .iter()
            .filter_map(|inst| match &inst.kind {
                InstKind::Commit { context, wrote, .. } => {
                    Some((self.i.resolve(context.name).to_owned(), *wrote))
                }
                _ => None,
            })
            .collect()
    }
}

fn solve(bodies: &[Body], contexts: &[&str]) -> Solved {
    let i = Interner::new();
    let mut pb = PolyBuilder::new();
    let mut functions: Vec<Function> = bodies
        .iter()
        .map(|body| Function {
            qref: QualifiedRef::root(i.intern(body.name)),
            kind: FnKind::Local(
                ParsedAst::Script(acvus_ast::parse_script(&i, body.source).expect("parse")),
                acvus_mir::graph::Inputs::FromReads,
            ),
            ty: TyTerm::Fn {
                params: vec![],
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: acvus_mir::ty::Effect::OPAQUE.into(),
                flows: acvus_mir::ty::Flows::Every.into(),
            },
        })
        .collect();
    let Externs {
        functions: std,
        types,
        ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), &i).expect("combine");
    functions.extend(std);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(
            contexts
                .iter()
                .map(|name| Context {
                    qref: QualifiedRef::root(i.intern(name)),
                    ty: pb.fresh_ty_var(),
                    init: None,
                })
                .collect(),
        ),
        types: Freeze::new(types),
        bindings: Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: Vec::new(),
    };
    let ext = extract::extract(&i, &graph);
    let inf = infer::infer(&i, &graph, &ext);
    let refused: Vec<String> = inf
        .errors()
        .into_iter()
        .flat_map(|(_, errs)| errs.iter().map(|e| e.display(&i).to_string()).collect::<Vec<_>>())
        .collect();
    assert!(refused.is_empty(), "the graph is refused: {refused:?}");
    let lowered = lower::lower(&i, &graph, &ext.view(), &inf);
    let refused: Vec<String> = lowered
        .errors
        .iter()
        .flat_map(|le| le.errors.iter().map(|e| e.display(&i).to_string()))
        .collect();
    assert!(refused.is_empty(), "lowering refused: {refused:?}");
    let types_by_context = inf
        .context_types
        .iter()
        .map(|(q, ty)| (i.resolve(q.name).to_owned(), ty.display(&i).to_string()))
        .collect();
    Solved {
        i,
        lowered,
        types_by_context,
    }
}

const INIT_LOG: Body = Body {
    name: "init",
    source: "@log = vec([]);",
};
const PUSH_LOG: Body = Body {
    name: "turn",
    source: r#"@log.push("x".to_string());"#,
};

#[test]
fn an_initializer_and_its_turn_solve_one_type_in_either_order() {
    for bodies in [[INIT_LOG, PUSH_LOG], [PUSH_LOG, INIT_LOG]] {
        let solved = solve(&bodies, &["log"]);
        assert_eq!(solved.ty("log"), "Vec<String>");
    }
}

#[test]
fn an_initializer_does_not_fetch_what_it_assigns_and_its_turn_does() {
    let solved = solve(&[INIT_LOG, PUSH_LOG], &["log"]);
    assert_eq!(solved.fetches("init"), Vec::<String>::new());
    assert_eq!(solved.fetches("turn"), ["log"]);
}

#[test]
fn a_caller_that_assigns_before_calling_fetches_only_after_the_call() {
    let solved = solve(
        &[
            Body {
                name: "read_x",
                source: "@x",
            },
            Body {
                name: "main",
                source: "@x = 3; read_x()",
            },
            Body {
                name: "call_only",
                source: "read_x()",
            },
        ],
        &["x"],
    );
    assert_eq!(solved.fetches("read_x"), ["x"]);
    assert_eq!(solved.fetches("main"), ["x"], "the fetch after the call");
    assert_eq!(solved.fetches("call_only"), Vec::<String>::new());
}

#[test]
fn two_stores_of_one_enum_context_solve_to_the_union_of_their_variants() {
    let solved = solve(
        &[
            Body {
                name: "init",
                source: "@mode = Mode::Idle;",
            },
            Body {
                name: "turn",
                source: "@mode = Mode::Busy;",
            },
        ],
        &["mode"],
    );
    assert_eq!(solved.ty("mode"), "Mode{Busy, Idle}");
}

#[test]
fn a_context_no_body_constrains_closes_to_never() {
    let solved = solve(
        &[Body {
            name: "init",
            source: "@xs = [];",
        }],
        &["xs"],
    );
    assert_eq!(solved.ty("xs"), "Array<!, 0>");
}

#[test]
fn a_commit_stores_only_where_the_body_may_have_written_since_the_fetch() {
    let solved = solve(
        &[
            Body {
                name: "set_x",
                source: "@x = 1;",
            },
            Body {
                name: "main",
                source: "let y = @x; set_x(); @x = y + 1; set_x(); let z = @x; z",
            },
        ],
        &["x"],
    );
    assert_eq!(
        solved.commits("main"),
        [
            ("x".to_owned(), false),
            ("x".to_owned(), true),
            ("x".to_owned(), false),
        ],
        "a read before the first call, the write before the second, nothing after it"
    );
    assert_eq!(solved.commits("set_x"), [("x".to_owned(), true)]);
}
