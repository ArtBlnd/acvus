pub mod builtins;
pub mod error;
pub mod extern_module;
pub mod hints;
pub mod ir;
pub mod lower;
pub mod printer;
pub mod ty;
pub mod typeck;

use std::collections::{HashMap, HashSet};

use acvus_ast::{Script, Template};

use crate::error::MirError;
use crate::extern_module::ExternRegistry;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::lower::Lowerer;
use crate::ty::Ty;
use crate::typeck::{TypeChecker, TypeMap};

/// Compile a parsed template into MIR.
///
/// - `template`: the parsed AST from `acvus_ast::parse()`.
/// - `context_types`: types for each `@name` context variable.
/// - `registry`: external function definitions.
///
/// Returns a `MirModule` and `HintTable`, or a list of errors.
pub fn compile(
    template: &Template,
    context_types: HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let context_names: HashSet<String> = context_types.keys().cloned().collect();
    let checker = TypeChecker::new(context_types, registry);
    let type_map = checker.check_template(template)?;
    let lowerer = Lowerer::new(type_map, context_names);
    let (module, hints) = lowerer.lower_template(template);
    Ok((module, hints))
}

/// Compile a parsed script into MIR (no type checking).
pub fn compile_script(script: &Script) -> (MirModule, HintTable) {
    let lowerer = Lowerer::new(TypeMap::new(), HashSet::new());
    lowerer.lower_script(script)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::extern_module::{ExternModule, ExternRegistry};

    fn compile_src(
        source: &str,
        context: HashMap<String, Ty>,
        registry: &ExternRegistry,
    ) -> Result<(MirModule, HintTable), Vec<MirError>> {
        let template = acvus_ast::parse(source).expect("parse failed");
        compile(&template, context, registry)
    }

    fn empty_registry() -> ExternRegistry {
        ExternRegistry::new()
    }

    #[test]
    fn integration_text_only() {
        let result = compile_src("hello world", HashMap::new(), &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_emit() {
        let result = compile_src(r#"{{ "hello" }}"#, HashMap::new(), &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_context_read_var_write() {
        // Variable write (no pre-declaration needed)
        let result = compile_src("{{ $count = 42 }}", HashMap::new(), &empty_registry());
        assert!(result.is_ok());

        // Context read
        let context = HashMap::from([("count".into(), Ty::Int)]);
        let result = compile_src("{{ @count | to_string }}", context, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_match_with_catch_all() {
        let context = HashMap::from([("name".into(), Ty::String)]);
        let result = compile_src(
            r#"{{ x = @name }}{{ x }}{{_}}default{{/}}"#,
            context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_variable_binding() {
        let context = HashMap::from([("name".into(), Ty::String)]);
        let result = compile_src(r#"{{ x = @name }}{{ x }}"#, context, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_extern_fn() {
        let mut module = ExternModule::new("test");
        module.add_fn("fetch_user", vec![Ty::Int], Ty::String, false);
        let mut registry = ExternRegistry::new();
        registry.register(&module);
        let result = compile_src(
            r#"{{ x = fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            HashMap::new(),
            &registry,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_pipe_with_lambda() {
        let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            r#"{{ x = @items | filter(x -> x != 0) }}{{ x | len | to_string }}{{_}}{{/}}"#,
            context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_field_access() {
        let context = HashMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])),
        )]);
        let result = compile_src("{{ @user.name }}", context, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_nested_match() {
        let context = HashMap::from([(
            "users".into(),
            Ty::List(Box::new(Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])))),
        )]);
        let result = compile_src(
            r#"{{ { name, } = @users }}{{ name }}{{/}}"#,
            context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_type_error_int_emit() {
        let result = compile_src("{{ 42 }}", HashMap::new(), &empty_registry());
        assert!(result.is_err());
    }

    #[test]
    fn integration_range_expression() {
        let result = compile_src(
            "{{ x in 0..10 }}{{ x | to_string }}{{/}}",
            HashMap::new(),
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_pattern() {
        let context = HashMap::from([(
            "data".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("value".into(), Ty::Int),
            ])),
        )]);
        let result = compile_src(
            r#"{{ { name, } = @data }}{{ name }}{{/}}"#,
            context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_multi_arm() {
        let context = HashMap::from([("role".into(), Ty::String)]);
        let result = compile_src(
            r#"{{ "admin" = @role }}admin page{{ "user" }}user page{{_}}guest{{/}}"#,
            context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_list_destructure() {
        let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}"#,
            context,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_concat() {
        let result = compile_src(
            r#"{{ "hello" + " " + "world" }}"#,
            HashMap::new(),
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_boolean_logic() {
        let result = compile_src("{{ true }}", HashMap::new(), &empty_registry());
        assert!(result.is_err());
    }

    #[test]
    fn script_single_expr() {
        let script = acvus_ast::parse_script("@data").unwrap();
        let (module, _) = compile_script(&script);
        let has_yield = module.main.insts.iter().any(|i| {
            matches!(&i.kind, crate::ir::InstKind::Yield(_))
        });
        assert!(has_yield);
    }

    #[test]
    fn script_bind_and_tail() {
        let script = acvus_ast::parse_script("x = @data; x").unwrap();
        let (module, _) = compile_script(&script);
        let has_context_load = module.main.insts.iter().any(|i| {
            matches!(&i.kind, crate::ir::InstKind::ContextLoad { .. })
        });
        let has_yield = module.main.insts.iter().any(|i| {
            matches!(&i.kind, crate::ir::InstKind::Yield(_))
        });
        assert!(has_context_load);
        assert!(has_yield);
    }

    #[test]
    fn script_trailing_semicolon_no_yield() {
        let script = acvus_ast::parse_script("x = @data;").unwrap();
        let (module, _) = compile_script(&script);
        let has_yield = module.main.insts.iter().any(|i| {
            matches!(&i.kind, crate::ir::InstKind::Yield(_))
        });
        assert!(!has_yield);
    }
}
