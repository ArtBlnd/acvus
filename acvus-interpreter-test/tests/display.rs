//! `core::display` at the contract (RFC-0070 rule 5, RFC-0071 rule 3): a
//! template's `{{ x }}` of a type that is not text appends through the
//! type's `display` instance, with the text `.to_string()` gives; a type
//! with no instance is refused at the tag; and `.to_string()` reaches the
//! generic over `display` at every type but `str`, whose owned copy is
//! `string::to_string`.

use acvus_extern::{ExternType, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Refusal, check_source, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Form {
    Template,
    Script,
}

/// A type the standard registries do not know, with a `display` instance of
/// its own.
#[derive(ExternType)]
#[extern_type(name = "Celsius")]
#[repr(transparent)]
struct Celsius(i64);

#[extern_fn(effect = pure)]
fn celsius(degrees: i64) -> Celsius {
    Celsius(degrees)
}

#[extern_fn(instance_of = acvus_extern::core::display, effect = pure)]
fn display_celsius(a: &Celsius, out: &mut String) {
    use std::fmt::Write;
    write!(out, "{}°C", a.0).expect("a String's fmt::Write does not fail");
}

fn celsius_registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "celsius", types: [Celsius], fns: [celsius, display_celsius] }
}

fn compile_and_run(source: &str, form: Form, opt: Opt) -> Result<Value, Refusal> {
    let i = Interner::new();
    let ast = match form {
        Form::Template => {
            ParsedAst::Template(acvus_ast::parse(&i, source).expect("the template parses"))
        }
        Form::Script => {
            ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the script parses"))
        }
    };
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(celsius_registry());
    let cr = check_source(
        &i,
        ast,
        &FxHashMap::default(),
        registries,
        Ty::String,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime
        .block_on(interp.execute())
        .expect("the seeds hold every context the run fetches"))
}

fn text_at_both_levels(source: &str, form: Form) -> String {
    let read = |opt| match compile_and_run(source, form, opt) {
        Ok(value) => {
            assert!(value.is_string(), "`{source}` at {opt:?}: {value:?}");
            // SAFETY: the witness is String.
            unsafe { value.as_str() }.to_owned()
        }
        Err(Refusal { messages, .. }) => {
            panic!("`{source}` at {opt:?} did not compile:\n  {}", messages.join("\n  "))
        }
    };
    let none = read(Opt::None);
    let full = read(Opt::Full);
    assert_eq!(none, full, "`{source}`: the two optimization levels differ");
    none
}

fn refusal_of_template(source: &str) -> String {
    match compile_and_run(source, Form::Template, Opt::Full) {
        Ok(value) => panic!("`{source}` compiled and ran to {value:?}"),
        Err(Refusal { messages, .. }) => messages.join("\n"),
    }
}

const DECIMAL: &str = "decimal(\"1.50\".to_string()).unwrap()";

/// A template line binding `d` to the decimal `1.50`, whose type the tag
/// that reads it knows.
const LET_DECIMAL: &str = "% if let Ok(d) = decimal(\"1.50\".to_string())\n";

/// `text` is what `.to_string()` gave at each former `core::to_string`
/// instance.
struct Shown {
    expr: String,
    text: &'static str,
}

fn every_display_type() -> Vec<Shown> {
    let shown = |expr: &str, text| Shown {
        expr: expr.to_string(),
        text,
    };
    vec![
        shown("42", "42"),
        shown("-1i8", "-1"),
        shown("-300i16", "-300"),
        shown("-70000i32", "-70000"),
        shown("200u8", "200"),
        shown("60000u16", "60000"),
        shown("4000000000u32", "4000000000"),
        shown("18446744073709551615u64", "18446744073709551615"),
        shown("1.5", "1.5"),
        shown("'c'", "c"),
        shown("true", "true"),
        shown("\"s\".to_string()", "s"),
    ]
}

#[test]
fn a_tag_of_a_type_with_an_instance_appends_what_to_string_gives() {
    for Shown { expr, text } in every_display_type() {
        assert_eq!(
            text_at_both_levels(&format!("<{{{{ {expr} }}}}>"), Form::Template),
            format!("<{text}>"),
            "{expr}"
        );
        assert_eq!(
            text_at_both_levels(&format!("<{{{{ ({expr}).to_string() }}}}>"), Form::Template),
            format!("<{text}>"),
            "{expr}"
        );
    }
}

#[test]
fn a_tag_of_a_decimal_appends_what_to_string_gives() {
    let source = format!("{LET_DECIMAL}<{{{{ d }}}}|{{{{ d.to_string() }}}}>\n% end");
    assert_eq!(text_at_both_levels(&source, Form::Template), "<1.50|1.50>\n");
}

#[test]
fn a_tag_lends_its_place_and_reads_through_a_reference() {
    let source = format!(
        "{LET_DECIMAL}% let n = 7i64\n% let r = &d\n\
         {{{{ d }}}} {{{{ n }}}} {{{{ &n }}}} {{{{ &d }}}} {{{{ r }}}} {{{{ d }}}} {{{{ n }}}}\n% end"
    );
    assert_eq!(
        text_at_both_levels(&source, Form::Template),
        "1.50 7 7 1.50 1.50 1.50 7\n"
    );
}

/// RFC-0071 rule 3: a tag of `&T` displays its `T`, an integer literal's
/// width settled by the solve, as `.to_string()` gives it.
#[test]
fn a_tag_of_a_reference_to_a_literal_s_integer_displays_it() {
    for source in [
        "% let n = 7\n% let r = &n\n<{{ r }}>",
        "% let n = 7\n<{{ &n }}>",
        "% let n = 7\n<{{ n.to_string() }}>",
    ] {
        assert_eq!(text_at_both_levels(source, Form::Template), "<7>", "{source}");
    }
}

#[test]
fn a_tag_of_text_appends_as_it_is() {
    assert_eq!(
        text_at_both_levels(
            "% let s = \"ab\".to_string()\n{{ s }}{{ \"cd\" }}{{ s }}",
            Form::Template
        ),
        "abcdab"
    );
}

struct Refused {
    source: &'static str,
    ty: &'static str,
}

#[test]
fn a_tag_of_a_type_with_no_instance_is_refused_naming_display() {
    for Refused { source, ty } in [
        Refused {
            source: "% let o = { a: 1, }\n{{ o }}",
            ty: "{a: i64}",
        },
        Refused {
            source: "% let e = Shade::Dark\n{{ e }}",
            ty: "Shade",
        },
        Refused {
            source: "% let f = |x| -> x + 1\n{{ f }}",
            ty: "Fn",
        },
    ] {
        let why = refusal_of_template(source);
        assert!(
            why.contains("has no instance of core::display for") && why.contains(ty),
            "`{source}`: {why}"
        );
    }
}

#[test]
fn to_string_gives_the_text_it_gave_at_every_former_instance_type() {
    for Shown { expr, text } in every_display_type() {
        assert_eq!(
            text_at_both_levels(&format!("({expr}).to_string()"), Form::Script),
            text,
            "{expr}"
        );
    }
    assert_eq!(
        text_at_both_levels(&format!("({DECIMAL}).to_string()"), Form::Script),
        "1.50"
    );
    assert_eq!(text_at_both_levels("\"lit\".to_string()", Form::Script), "lit");
    assert_eq!(text_at_both_levels("let s = \"x\"; s.to_string()", Form::Script), "x");
}

/// A type's text has one source: an extension type that declares `display`
/// has `.to_string()` through the generic, and a tag appends the same text.
#[test]
fn an_extension_type_s_display_instance_is_its_to_string() {
    assert_eq!(
        text_at_both_levels("celsius(21).to_string()", Form::Script),
        "21°C"
    );
    assert_eq!(
        text_at_both_levels("% let c = celsius(-4)\n{{ c }}|{{ c.to_string() }}", Form::Template),
        "-4°C|-4°C"
    );
}
