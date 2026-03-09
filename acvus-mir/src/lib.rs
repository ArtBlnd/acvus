pub mod builtins;
pub mod error;
pub mod extern_module;
pub mod hints;
pub mod ir;
pub mod lower;
pub mod printer;
pub mod ty;
pub mod typeck;
pub mod user_type;
pub mod variant;

use std::collections::HashMap;

use acvus_ast::{Script, Template};
use acvus_utils::{Astr, Interner};

use crate::error::MirError;
use crate::extern_module::ExternRegistry;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::lower::Lowerer;
use crate::ty::Ty;
use crate::typeck::TypeChecker;
use crate::user_type::UserTypeRegistry;

/// Compile a parsed template into MIR.
///
/// - `template`: the parsed AST from `acvus_ast::parse()`.
/// - `context_types`: types for each `@name` context variable.
/// - `registry`: external function definitions.
/// - `user_types`: user-defined enum types.
///
/// Returns a `MirModule` and `HintTable`, or a list of errors.
pub fn compile(
    interner: &Interner,
    template: &Template,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let checker = TypeChecker::new(interner, context_types, registry, user_types);
    let (type_map, variant_registry) = checker.check_template(template)?;
    let lowerer = Lowerer::new(interner, type_map, variant_registry, registry);
    let (module, hints) = lowerer.lower_template(template);
    Ok((module, hints))
}

/// Compile a parsed script with type checking. Returns the MIR module, hint table, and the tail expression type.
pub fn compile_script(
    interner: &Interner,
    script: &Script,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    compile_script_with_hint(interner, script, context_types, registry, user_types, None)
}

pub fn compile_script_with_hint(
    interner: &Interner,
    script: &Script,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let checker = TypeChecker::new(interner, context_types, registry, user_types);
    let (type_map, tail_ty, variant_registry) =
        checker.check_script_with_hint(script, expected_tail)?;
    let lowerer = Lowerer::new(interner, type_map, variant_registry, registry);
    let (module, hints) = lowerer.lower_script(script);
    Ok((module, hints, tail_ty))
}

/// Compile a template in analysis mode: unknown `@context` refs get fresh
/// type variables instead of errors, enabling partial type inference.
pub fn compile_analysis(
    interner: &Interner,
    template: &Template,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let checker = TypeChecker::new(interner, context_types, registry, user_types).with_analysis_mode();
    let (type_map, variant_registry) = checker.check_template(template)?;
    let lowerer = Lowerer::new(interner, type_map, variant_registry, registry);
    let (module, hints) = lowerer.lower_template(template);
    Ok((module, hints))
}

/// Compile a script in analysis mode.
pub fn compile_script_analysis(
    interner: &Interner,
    script: &Script,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    compile_script_analysis_with_tail(interner, script, context_types, registry, user_types, None)
}

/// Compile a script in analysis mode with an expected tail type hint.
pub fn compile_script_analysis_with_tail(
    interner: &Interner,
    script: &Script,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
    user_types: &UserTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let checker = TypeChecker::new(interner, context_types, registry, user_types).with_analysis_mode();
    let (type_map, tail_ty, variant_registry) =
        checker.check_script_with_hint(script, expected_tail)?;
    let lowerer = Lowerer::new(interner, type_map, variant_registry, registry);
    let (module, hints) = lowerer.lower_script(script);
    Ok((module, hints, tail_ty))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::extern_module::{ExternModule, ExternRegistry};

    fn compile_src(
        interner: &Interner,
        source: &str,
        context: &HashMap<Astr, Ty>,
        registry: &ExternRegistry,
    ) -> Result<(MirModule, HintTable), Vec<MirError>> {
        let template = acvus_ast::parse(interner, source).expect("parse failed");
        compile(interner, &template, context, registry, &UserTypeRegistry::new())
    }

    fn empty_registry() -> ExternRegistry {
        ExternRegistry::new()
    }

    #[test]
    fn integration_text_only() {
        let i = Interner::new();
        let result = compile_src(&i, "hello world", &HashMap::new(), &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_emit() {
        let i = Interner::new();
        let result = compile_src(&i, r#"{{ "hello" }}"#, &HashMap::new(), &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_context_read_var_write() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ $count = 42 }}", &HashMap::new(), &empty_registry());
        assert!(result.is_ok());

        let context = HashMap::from([(i.intern("count"), Ty::Int)]);
        let result = compile_src(&i, "{{ @count | to_string }}", &context, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_match_with_catch_all() {
        let i = Interner::new();
        let context = HashMap::from([(i.intern("name"), Ty::String)]);
        let result = compile_src(
            &i,
            r#"{{ x = @name }}{{ x }}{{_}}default{{/}}"#,
            &context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_variable_binding() {
        let i = Interner::new();
        let context = HashMap::from([(i.intern("name"), Ty::String)]);
        let result = compile_src(&i, r#"{{ x = @name }}{{ x }}"#, &context, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_extern_fn() {
        let i = Interner::new();
        let mut module = ExternModule::new(i.intern("test"));
        module.add_fn(i.intern("fetch_user"), vec![Ty::Int], Ty::String, false);
        let mut registry = ExternRegistry::new();
        registry.register(&module);
        let result = compile_src(
            &i,
            r#"{{ x = fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            &HashMap::new(),
            &registry,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_pipe_with_lambda() {
        let i = Interner::new();
        let context = HashMap::from([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            &i,
            r#"{{ x = @items | filter(x -> x != 0) }}{{ x | len | to_string }}{{_}}{{/}}"#,
            &context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_field_access() {
        let i = Interner::new();
        let context = HashMap::from([(
            i.intern("user"),
            Ty::Object(HashMap::from([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])),
        )]);
        let result = compile_src(&i, "{{ @user.name }}", &context, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_nested_match() {
        let i = Interner::new();
        let context = HashMap::from([(
            i.intern("users"),
            Ty::List(Box::new(Ty::Object(HashMap::from([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])))),
        )]);
        let result = compile_src(
            &i,
            r#"{{ { name, } = @users }}{{ name }}{{/}}"#,
            &context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_type_error_int_emit() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ 42 }}", &HashMap::new(), &empty_registry());
        assert!(result.is_err());
    }

    #[test]
    fn integration_range_expression() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            "{{ x in 0..10 }}{{ x | to_string }}{{/}}",
            &HashMap::new(),
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_pattern() {
        let i = Interner::new();
        let context = HashMap::from([(
            i.intern("data"),
            Ty::Object(HashMap::from([
                (i.intern("name"), Ty::String),
                (i.intern("value"), Ty::Int),
            ])),
        )]);
        let result = compile_src(
            &i,
            r#"{{ { name, } = @data }}{{ name }}{{/}}"#,
            &context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_multi_arm() {
        let i = Interner::new();
        let context = HashMap::from([(i.intern("role"), Ty::String)]);
        let result = compile_src(
            &i,
            r#"{{ "admin" = @role }}admin page{{ "user" }}user page{{_}}guest{{/}}"#,
            &context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_list_destructure() {
        let i = Interner::new();
        let context = HashMap::from([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            &i,
            r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}"#,
            &context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_concat() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            r#"{{ "hello" + " " + "world" }}"#,
            &HashMap::new(),
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_boolean_logic() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ true }}", &HashMap::new(), &empty_registry());
        assert!(result.is_err());
    }

    fn compile_script_test(source: &str) -> MirModule {
        let i = Interner::new();
        let script = acvus_ast::parse_script(&i, source).unwrap();
        let ctx = HashMap::from([(i.intern("data"), Ty::String)]);
        let (module, _, _) =
            compile_script(&i, &script, &ctx, &empty_registry(), &UserTypeRegistry::new()).unwrap();
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
