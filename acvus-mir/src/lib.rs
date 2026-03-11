pub mod builtins;
pub mod error;
pub mod hints;
pub mod ir;
pub mod lower;
pub mod printer;
pub mod ty;
pub mod typeck;
pub mod user_type;
pub mod variant;

use acvus_ast::{Script, Template};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::lower::Lowerer;
use crate::ty::Ty;
use crate::typeck::TypeChecker;

/// Compile a parsed template into MIR.
///
/// - `template`: the parsed AST from `acvus_ast::parse()`.
/// - `context_types`: types for each `@name` context variable.
///
/// Returns a `MirModule` and `HintTable`, or a list of errors.
pub fn compile(
    interner: &Interner,
    template: &Template,
    context_types: &FxHashMap<Astr, Ty>,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let checker = TypeChecker::new(interner, context_types);
    let type_map = checker.check_template(template)?;
    let lowerer = Lowerer::new(interner, type_map);
    let (module, hints) = lowerer.lower_template(template);
    Ok((module, hints))
}

/// Compile a parsed script with type checking. Returns the MIR module, hint table, and the tail expression type.
pub fn compile_script(
    interner: &Interner,
    script: &Script,
    context_types: &FxHashMap<Astr, Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    compile_script_with_hint(interner, script, context_types, None)
}

pub fn compile_script_with_hint(
    interner: &Interner,
    script: &Script,
    context_types: &FxHashMap<Astr, Ty>,
    expected_tail: Option<&Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let checker = TypeChecker::new(interner, context_types);
    let (type_map, tail_ty) =
        checker.check_script_with_hint(script, expected_tail)?;
    let lowerer = Lowerer::new(interner, type_map);
    let (module, hints) = lowerer.lower_script(script);
    Ok((module, hints, tail_ty))
}

/// Compile a template in analysis mode: unknown `@context` refs get fresh
/// type variables instead of errors, enabling partial type inference.
pub fn compile_analysis(
    interner: &Interner,
    template: &Template,
    context_types: &FxHashMap<Astr, Ty>,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let checker =
        TypeChecker::new(interner, context_types).with_analysis_mode();
    let type_map = checker.check_template(template)?;
    let lowerer = Lowerer::new(interner, type_map);
    let (module, hints) = lowerer.lower_template(template);
    Ok((module, hints))
}

/// Like `compile_analysis`, but always returns a module even when type errors exist.
/// Errors are returned alongside the module so callers can still extract context keys.
pub fn compile_analysis_partial(
    interner: &Interner,
    template: &Template,
    context_types: &FxHashMap<Astr, Ty>,
) -> (MirModule, HintTable, Vec<MirError>) {
    let checker =
        TypeChecker::new(interner, context_types).with_analysis_mode();
    let (type_map, errors) = checker.check_template_partial(template);
    let lowerer = Lowerer::new(interner, type_map);
    let (module, hints) = lowerer.lower_template(template);
    (module, hints, errors)
}

/// Compile a script in analysis mode.
pub fn compile_script_analysis(
    interner: &Interner,
    script: &Script,
    context_types: &FxHashMap<Astr, Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    compile_script_analysis_with_tail(interner, script, context_types, None)
}

/// Compile a script in analysis mode with an expected tail type hint.
pub fn compile_script_analysis_with_tail(
    interner: &Interner,
    script: &Script,
    context_types: &FxHashMap<Astr, Ty>,
    expected_tail: Option<&Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let checker =
        TypeChecker::new(interner, context_types).with_analysis_mode();
    let (type_map, tail_ty) =
        checker.check_script_with_hint(script, expected_tail)?;
    let lowerer = Lowerer::new(interner, type_map);
    let (module, hints) = lowerer.lower_script(script);
    Ok((module, hints, tail_ty))
}

/// Like `compile_script_analysis_with_tail`, but always returns a module even when type errors exist.
pub fn compile_script_analysis_with_tail_partial(
    interner: &Interner,
    script: &Script,
    context_types: &FxHashMap<Astr, Ty>,
    expected_tail: Option<&Ty>,
) -> (MirModule, HintTable, Ty, Vec<MirError>) {
    let checker =
        TypeChecker::new(interner, context_types).with_analysis_mode();
    let (type_map, tail_ty, errors) =
        checker.check_script_with_hint_partial(script, expected_tail);
    let lowerer = Lowerer::new(interner, type_map);
    let (module, hints) = lowerer.lower_script(script);
    (module, hints, tail_ty, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_src(
        interner: &Interner,
        source: &str,
        context: &FxHashMap<Astr, Ty>,
    ) -> Result<(MirModule, HintTable), Vec<MirError>> {
        let template = acvus_ast::parse(interner, source).expect("parse failed");
        compile(interner, &template, context)
    }

    #[test]
    fn integration_text_only() {
        let i = Interner::new();
        let result = compile_src(&i, "hello world", &FxHashMap::default());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_emit() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            r#"{{ "hello" }}"#,
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_context_read_var_write() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            "{{ $count = 42 }}",
            &FxHashMap::default(),
        );
        assert!(result.is_ok());

        let context = FxHashMap::from_iter([(i.intern("count"), Ty::Int)]);
        let result = compile_src(&i, "{{ @count | to_string }}", &context);
        assert!(result.is_ok());
    }

    #[test]
    fn integration_match_with_catch_all() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let result = compile_src(
            &i,
            r#"{{ x = @name }}{{ x }}{{_}}default{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_variable_binding() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let result = compile_src(&i, r#"{{ x = @name }}{{ x }}"#, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn integration_extern_fn() {
        let i = Interner::new();
        // Extern functions are now provided via context_types with Ty::Fn { is_extern: true }
        let context = FxHashMap::from_iter([(
            i.intern("fetch_user"),
            Ty::Fn {
                params: vec![Ty::Int],
                ret: Box::new(Ty::String),
                is_extern: true,
            },
        )]);
        let result = compile_src(
            &i,
            r#"{{ x = @fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_pipe_with_lambda() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            &i,
            r#"{{ x = @items | filter(x -> x != 0) }}{{ x | len | to_string }}{{_}}{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_field_access() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("user"),
            Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])),
        )]);
        let result = compile_src(&i, "{{ @user.name }}", &context);
        assert!(result.is_ok());
    }

    #[test]
    fn integration_nested_match() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("users"),
            Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])))),
        )]);
        let result = compile_src(
            &i,
            r#"{{ { name, } = @users }}{{ name }}{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_type_error_int_emit() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ 42 }}", &FxHashMap::default());
        assert!(result.is_err());
    }

    #[test]
    fn integration_range_expression() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            "{{ x in 0..10 }}{{ x | to_string }}{{/}}",
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_pattern() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("data"),
            Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("value"), Ty::Int),
            ])),
        )]);
        let result = compile_src(
            &i,
            r#"{{ { name, } = @data }}{{ name }}{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_multi_arm() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("role"), Ty::String)]);
        let result = compile_src(
            &i,
            r#"{{ "admin" = @role }}admin page{{ "user" }}user page{{_}}guest{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_list_destructure() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            &i,
            r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}"#,
            &context,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_concat() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            r#"{{ "hello" + " " + "world" }}"#,
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_boolean_logic() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ true }}", &FxHashMap::default());
        assert!(result.is_err());
    }

    fn compile_script_test(source: &str) -> MirModule {
        let i = Interner::new();
        let script = acvus_ast::parse_script(&i, source).unwrap();
        let ctx = FxHashMap::from_iter([(i.intern("data"), Ty::String)]);
        let (module, _, _) = compile_script(&i, &script, &ctx)
            .unwrap();
        module
    }

    #[test]
    fn script_single_expr() {
        let module = compile_script_test("@data");
        let has_yield = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, crate::ir::InstKind::Yield(_)));
        assert!(has_yield);
    }

    #[test]
    fn script_bind_and_tail() {
        let module = compile_script_test("x = @data; x");
        let has_context_load = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, crate::ir::InstKind::ContextLoad { .. }));
        let has_yield = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, crate::ir::InstKind::Yield(_)));
        assert!(has_context_load);
        assert!(has_yield);
    }

    #[test]
    fn script_trailing_semicolon_no_yield() {
        let module = compile_script_test("x = @data;");
        let has_yield = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, crate::ir::InstKind::Yield(_)));
        assert!(!has_yield);
    }
}
