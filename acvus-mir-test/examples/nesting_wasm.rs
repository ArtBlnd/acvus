//! RFC-0106 rule 2: the stack the compiler takes on `wasm32` for a source
//! nested to a given depth, for each kind of level the native
//! `nesting_bound` test compiles. `examples/nesting_wasm.mjs` runs the module
//! under node and reads the stack each compile took; docs/nesting.md names the
//! command and records the numbers `NESTING_MAX` was chosen from.

use acvus_ast::{NESTING_MAX, ParseError};
use acvus_extern::{Externs, TypesOnly};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Access, Bindings, CompilationGraph, FnKind, Function, Inputs, ParsedAst, QualifiedRef,
    extract, infer, lower, optimize,
};
use acvus_mir::laws::LawTable;
use acvus_mir::ty::{Effect, Flows, PolyBuilder, Ty, TyTerm, lift_declaration};
use acvus_utils::{Freeze, Interner};

#[derive(Clone, Copy)]
enum Form {
    Script,
    Template,
}

#[repr(u32)]
enum Status {
    Compiled = 0,
    ParseRefused = 1,
    CheckRefused = 2,
    NoSuchKind = 3,
}

const KINDS: [(Form, fn(usize) -> String); 7] = [
    (Form::Script, parens),
    (Form::Script, chain),
    (Form::Script, blocks),
    (Form::Script, statements),
    (Form::Script, lambdas),
    (Form::Script, patterns),
    (Form::Template, sections),
];

fn parens(levels: usize) -> String {
    format!("{}1{}", "(".repeat(levels - 1), ")".repeat(levels - 1))
}

fn chain(levels: usize) -> String {
    format!("1{}", " + 1".repeat(levels - 1))
}

fn blocks(levels: usize) -> String {
    format!("{}1{}", "{ ".repeat(levels - 1), " }".repeat(levels - 1))
}

fn statements(levels: usize) -> String {
    let (innermost, below) = match levels % 2 == 0 {
        true => ("(1)", 2),
        false => ("1", 1),
    };
    let count = (levels - below) / 2;
    format!("{}{innermost}{}", "{ let a = ".repeat(count), "; a }".repeat(count))
}

fn lambdas(levels: usize) -> String {
    let count = levels - 2;
    format!("let f = {}x; f{}", "|x| -> ".repeat(count), "(1)".repeat(count))
}

fn patterns(levels: usize) -> String {
    let count = levels - 2;
    let some = |inner: &str| format!("{}{inner}{}", "Some(".repeat(count), ")".repeat(count));
    format!("match {} {{ {} => x, _ => 0 }}", some("1"), some("x"))
}

fn sections(levels: usize) -> String {
    let count = (levels - 2) / 2;
    let line = match levels % 2 == 0 {
        true => "x\n",
        false => "{{ (\"x\") }}\n",
    };
    format!("{}{line}{}", "% if true\n".repeat(count), "% end\n".repeat(count))
}

fn parsed(interner: &Interner, form: Form, source: &str) -> Result<ParsedAst, Vec<ParseError>> {
    match form {
        Form::Script => acvus_ast::parse_script(interner, source)
            .map(ParsedAst::Script)
            .map_err(|recovered| recovered.errors),
        Form::Template => acvus_ast::parse_template(interner, source)
            .map(ParsedAst::Template)
            .map_err(|recovered| recovered.errors),
    }
}

fn compiled(form: Form, source: &str) -> Status {
    let interner = Interner::new();
    let Ok(ast) = parsed(&interner, form, source) else {
        return Status::ParseRefused;
    };
    let ret = match form {
        Form::Script => Ty::I64,
        Form::Template => Ty::String,
    };
    let mut pb = PolyBuilder::new();
    let entry = QualifiedRef::root(interner.intern("main"));
    let mut functions = vec![Function {
        qref: entry,
        kind: FnKind::Local(ast, Inputs::FromReads),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(lift_declaration(&ret, &mut pb)),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: Flows::Every.into(),
        },
    }];
    let Externs {
        functions: externs,
        types,
        ..
    } = Externs::combine(acvus_ext::std_registries::<TypesOnly>(), &interner)
        .expect("the standard registries combine");
    functions.extend(externs);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(types),
        bindings: Bindings::default(),
        access: Access::Sync,
        entries: vec![entry],
    };
    let extracted = extract::extract(&interner, &graph);
    let inferred = infer::infer(&interner, &graph, &extracted);
    let refused_in_check = inferred.outcomes.values().any(|outcome| {
        matches!(outcome, infer::FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty())
    });
    if refused_in_check {
        return Status::CheckRefused;
    }
    let lowered = lower::lower(&interner, &graph, &extracted.view(), &inferred);
    if !lowered.errors.is_empty() {
        return Status::CheckRefused;
    }
    let laws = LawTable::of(graph.functions.iter(), &graph.types);
    let optimized = optimize::optimize(&interner, &laws, lowered.modules, Opt::Full);
    match optimized.errors.is_empty() {
        true => Status::Compiled,
        false => Status::CheckRefused,
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn stack_pointer() -> usize {
    let local = 0u8;
    std::hint::black_box(std::ptr::addr_of!(local)).addr()
}

#[unsafe(no_mangle)]
pub extern "C" fn nesting_max() -> u32 {
    NESTING_MAX
}

#[unsafe(no_mangle)]
pub extern "C" fn kind_count() -> u32 {
    u32::try_from(KINDS.len()).expect("seven kinds")
}

#[unsafe(no_mangle)]
pub extern "C" fn compile(kind: u32, levels: u32) -> u32 {
    let kind = usize::try_from(kind).expect("a u32 fits a usize on every target this builds for");
    let Some((form, source)) = KINDS.get(kind) else {
        return Status::NoSuchKind as u32;
    };
    let levels =
        usize::try_from(levels).expect("a u32 fits a usize on every target this builds for");
    compiled(*form, &source(levels)) as u32
}
