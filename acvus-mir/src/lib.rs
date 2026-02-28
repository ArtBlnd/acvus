pub mod builtins;
pub mod error;
pub mod hints;
pub mod ir;
pub mod lower;
pub mod printer;
pub mod ty;
pub mod typeck;

use std::collections::{HashMap, HashSet};

use acvus_ast::Template;

use crate::error::MirError;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::lower::Lowerer;
use crate::ty::Ty;
use crate::typeck::TypeChecker;

/// Compile a parsed template into MIR.
///
/// - `template`: the parsed AST from `acvus_ast::parse()`.
/// - `storage_types`: types for each `$name` storage variable.
/// - `extern_fns`: signatures for external (host-provided) functions.
///
/// Returns a `MirModule` and `HintTable`, or a list of errors.
pub fn compile(
    template: &Template,
    storage_types: HashMap<String, Ty>,
    extern_fns: HashMap<String, (Vec<Ty>, Ty)>,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let storage_names: HashSet<String> = storage_types.keys().cloned().collect();
    let checker = TypeChecker::new(storage_types, extern_fns);
    let type_map = checker.check_template(template)?;
    let lowerer = Lowerer::new(type_map, storage_names);
    let (module, hints) = lowerer.lower_template(template);
    Ok((module, hints))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn compile_src(
        source: &str,
        storage: HashMap<String, Ty>,
        externs: HashMap<String, (Vec<Ty>, Ty)>,
    ) -> Result<(MirModule, HintTable), Vec<MirError>> {
        let template = acvus_ast::parse(source).expect("parse failed");
        compile(&template, storage, externs)
    }

    #[test]
    fn integration_text_only() {
        let result = compile_src("hello world", HashMap::new(), HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_emit() {
        let result = compile_src(r#"{{ "hello" }}"#, HashMap::new(), HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_storage_read_write() {
        let storage = HashMap::from([("count".into(), Ty::Int)]);
        let result = compile_src("{{ $count = 42 }}", storage.clone(), HashMap::new());
        assert!(result.is_ok());

        let result = compile_src("{{ $count | to_string }}", storage, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_match_with_catch_all() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        let result = compile_src(
            r#"{{ x = $name }}{{ x }}{{_}}default{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_variable_binding() {
        // Variable bindings are body-less and define in the current scope.
        let storage = HashMap::from([("name".into(), Ty::String)]);
        let result = compile_src(r#"{{ x = $name }}{{ x }}"#, storage, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_extern_fn() {
        let externs =
            HashMap::from([("fetch_user".into(), (vec![Ty::Int], Ty::String))]);
        let result = compile_src(
            r#"{{ x = fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            HashMap::new(),
            externs,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_pipe_with_lambda() {
        let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            r#"{{ x = $items | filter(x -> x != 0) }}{{ x | to_string }}{{_}}{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_field_access() {
        let storage = HashMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])),
        )]);
        let result = compile_src("{{ $user.name }}", storage, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_nested_match() {
        // Use a non-binding pattern to trigger iteration match block.
        let storage = HashMap::from([(
            "users".into(),
            Ty::List(Box::new(Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])))),
        )]);
        let result = compile_src(
            r#"{{ { name, } = $users }}{{ name }}{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_type_error_int_emit() {
        let result = compile_src("{{ 42 }}", HashMap::new(), HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn integration_range_expression() {
        let result = compile_src(
            "{{ x = 0..10 }}{{ x | to_string }}{{_}}{{/}}",
            HashMap::new(),
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_pattern() {
        let storage = HashMap::from([(
            "data".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("value".into(), Ty::Int),
            ])),
        )]);
        let result = compile_src(
            r#"{{ { name, } = $data }}{{ name }}{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_multi_arm() {
        let storage = HashMap::from([("role".into(), Ty::String)]);
        // Bare literal in match block body is a continuation arm.
        let result = compile_src(
            r#"{{ "admin" = $role }}admin page{{ "user" }}user page{{_}}guest{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_list_destructure() {
        let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let result = compile_src(
            r#"{{ [a, b, ..] = $items }}{{ a | to_string }}{{_}}{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_concat() {
        let result = compile_src(
            r#"{{ "hello" + " " + "world" }}"#,
            HashMap::new(),
            HashMap::new(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_boolean_logic() {
        // Bool emit should fail (not String).
        let result = compile_src("{{ true }}", HashMap::new(), HashMap::new());
        assert!(result.is_err());
    }
}
