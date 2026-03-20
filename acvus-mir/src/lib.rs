pub mod analysis;
pub mod builtins;
pub mod context_registry;
pub mod error;
pub mod graph;
pub mod hints;
pub mod ir;
pub mod lower;
pub mod optimize;
pub mod pass;
pub mod printer;
pub mod ty;
pub mod ser_ty;
pub mod typeck;
pub mod validate;
pub mod variant;

pub use pass::AnalysisPass;
pub use typeck::{TypeResolution, Unchecked, Checked, check_completeness};

use acvus_ast::{Script, Template};
use acvus_utils::Interner;

use crate::context_registry::ContextTypeRegistry;
use crate::error::MirError;
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::lower::Lowerer;
use crate::ty::{Ty, TySubst};
use crate::typeck::TypeChecker;

/// Compile a parsed template into MIR.
///
/// - `template`: the parsed AST from `acvus_ast::parse()`.
/// - `registry`: context type registry providing types for each `@name` context variable.
///
/// Returns a `MirModule` and `HintTable`, or a list of errors.
pub fn compile(
    interner: &Interner,
    template: &Template,
    registry: &ContextTypeRegistry,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let mut subst = TySubst::new();
    let checker = TypeChecker::new(interner, registry.merged(), &mut subst);
    let (type_map, builtin_map, coercion_map) = checker.check_template(template)?;
    let lowerer = Lowerer::new(interner, type_map, builtin_map, coercion_map);
    let (module, hints) = lowerer.lower_template(template);
    let validation_errors = validate::validate(&module);
    if !validation_errors.is_empty() {
        return Err(validation_errors.into_iter().map(|e| e.into_mir_error()).collect());
    }
    Ok((module, hints))
}

/// Compile a parsed script with type checking. Returns the MIR module, hint table, and the tail expression type.
pub fn compile_script(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    compile_script_with_hint(interner, script, registry, None)
}

pub fn compile_script_with_hint(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let mut subst = TySubst::new();
    compile_script_with_hint_subst(interner, script, registry, expected_tail, &mut subst)
}

/// Like `compile_script_with_hint`, but uses an externally provided `TySubst`.
/// This allows sharing origin/type-variable state across multiple compilations
/// (e.g. initial_value and bind script for the same node).
pub fn compile_script_with_hint_subst(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
    subst: &mut TySubst,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let checker = TypeChecker::new(interner, registry.merged(), subst);
    let (type_map, builtin_map, coercion_map, tail_ty) =
        checker.check_script_with_hint(script, expected_tail)?;
    let lowerer = Lowerer::new(interner, type_map, builtin_map, coercion_map);
    let (module, hints) = lowerer.lower_script(script);
    let validation_errors = validate::validate(&module);
    if !validation_errors.is_empty() {
        return Err(validation_errors.into_iter().map(|e| e.into_mir_error()).collect());
    }
    Ok((module, hints, tail_ty))
}

// ── New layered API ──────────────────────────────────────────────────
//
// Layer 1: typecheck only → TypeResolution<Unchecked>
// Bridge:  check_completeness → TypeResolution<Checked>
// Layer 2: lower → MirModule
//
// These replace the monolithic compile_* functions above.

/// Layer 1: Type check a script. Returns TypeResolution<Unchecked>.
/// Unresolved type variables are allowed — call check_completeness after
/// all SCC units are typechecked.
pub fn typecheck_script(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
    subst: &mut TySubst,
) -> Result<TypeResolution<Unchecked>, Vec<MirError>> {
    let checker = TypeChecker::new(interner, registry.merged(), subst);
    checker.check_script_to_resolution(script, expected_tail)
}

/// Layer 1: Type check a template. Returns TypeResolution<Unchecked>.
pub fn typecheck_template(
    interner: &Interner,
    template: &Template,
    registry: &ContextTypeRegistry,
    subst: &mut TySubst,
) -> Result<TypeResolution<Unchecked>, Vec<MirError>> {
    let checker = TypeChecker::new(interner, registry.merged(), subst);
    checker.check_template_to_resolution(template)
}

/// Layer 2: Lower a checked script to MIR. Only accepts TypeResolution<Checked>.
pub fn lower_checked_script(
    interner: &Interner,
    script: &Script,
    resolution: TypeResolution<Checked>,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let lowerer = Lowerer::new(interner, resolution.type_map, resolution.builtin_map, resolution.coercion_map);
    let (module, hints) = lowerer.lower_script(script);
    let validation_errors = validate::validate(&module);
    if !validation_errors.is_empty() {
        return Err(validation_errors.into_iter().map(|e| e.into_mir_error()).collect());
    }
    Ok((module, hints))
}

/// Layer 2: Lower a checked template to MIR. Only accepts TypeResolution<Checked>.
pub fn lower_checked_template(
    interner: &Interner,
    template: &Template,
    resolution: TypeResolution<Checked>,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let lowerer = Lowerer::new(interner, resolution.type_map, resolution.builtin_map, resolution.coercion_map);
    let (module, hints) = lowerer.lower_template(template);
    let validation_errors = validate::validate(&module);
    if !validation_errors.is_empty() {
        return Err(validation_errors.into_iter().map(|e| e.into_mir_error()).collect());
    }
    Ok((module, hints))
}

// ── Legacy API (to be removed in Step 6) ────────────────────────────

/// Compile a template in analysis mode: unknown `@context` refs get fresh
/// type variables instead of errors, enabling partial type inference.
pub fn compile_analysis(
    interner: &Interner,
    template: &Template,
    registry: &ContextTypeRegistry,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let mut subst = TySubst::new();
    let checker =
        TypeChecker::new(interner, registry.merged(), &mut subst).with_analysis_mode();
    let (type_map, builtin_map, coercion_map) = checker.check_template(template)?;
    let lowerer = Lowerer::new(interner, type_map, builtin_map, coercion_map);
    let (module, hints) = lowerer.lower_template(template);
    let validation_errors = validate::validate(&module);
    if !validation_errors.is_empty() {
        return Err(validation_errors.into_iter().map(|e| e.into_mir_error()).collect());
    }
    Ok((module, hints))
}

/// Like `compile_analysis`, but always returns a module even when type errors exist.
/// Errors are returned alongside the module so callers can still extract context keys.
pub fn compile_analysis_partial(
    interner: &Interner,
    template: &Template,
    registry: &ContextTypeRegistry,
) -> (MirModule, HintTable, Vec<MirError>) {
    let mut subst = TySubst::new();
    let checker =
        TypeChecker::new(interner, registry.merged(), &mut subst).with_analysis_mode();
    let (type_map, builtin_map, coercion_map, mut errors) = checker.check_template_partial(template);
    let lowerer = Lowerer::new(interner, type_map, builtin_map, coercion_map);
    let (module, hints) = lowerer.lower_template(template);
    let validation_errors = validate::validate(&module);
    errors.extend(validation_errors.into_iter().map(|e| e.into_mir_error()));
    (module, hints, errors)
}

/// Compile a script in analysis mode.
pub fn compile_script_analysis(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    compile_script_analysis_with_tail(interner, script, registry, None)
}

/// Compile a script in analysis mode with an expected tail type hint.
pub fn compile_script_analysis_with_tail(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
    let mut subst = TySubst::new();
    let checker =
        TypeChecker::new(interner, registry.merged(), &mut subst).with_analysis_mode();
    let (type_map, builtin_map, coercion_map, tail_ty) =
        checker.check_script_with_hint(script, expected_tail)?;
    let lowerer = Lowerer::new(interner, type_map, builtin_map, coercion_map);
    let (module, hints) = lowerer.lower_script(script);
    let validation_errors = validate::validate(&module);
    if !validation_errors.is_empty() {
        return Err(validation_errors.into_iter().map(|e| e.into_mir_error()).collect());
    }
    Ok((module, hints, tail_ty))
}

/// Like `compile_script_analysis_with_tail`, but always returns a module even when type errors exist.
pub fn compile_script_analysis_with_tail_partial(
    interner: &Interner,
    script: &Script,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> (MirModule, HintTable, Ty, Vec<MirError>) {
    let mut subst = TySubst::new();
    let checker =
        TypeChecker::new(interner, registry.merged(), &mut subst).with_analysis_mode();
    let (type_map, builtin_map, coercion_map, tail_ty, mut errors) =
        checker.check_script_with_hint_partial(script, expected_tail);
    let lowerer = Lowerer::new(interner, type_map, builtin_map, coercion_map);
    let (module, hints) = lowerer.lower_script(script);
    let validation_errors = validate::validate(&module);
    errors.extend(validation_errors.into_iter().map(|e| e.into_mir_error()));
    (module, hints, tail_ty, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_registry::ContextTypeRegistry;
    use crate::ty::{Effect, FnKind};
    use rustc_hash::FxHashMap;

    fn compile_src(
        interner: &Interner,
        source: &str,
        registry: &ContextTypeRegistry,
    ) -> Result<(MirModule, HintTable), Vec<MirError>> {
        let template = acvus_ast::parse(interner, source).expect("parse failed");
        let mut subst = TySubst::new();
        let unchecked = typecheck_template(interner, &template, registry, &mut subst)?;
        let checked = check_completeness(unchecked, &subst)?;
        lower_checked_template(interner, &template, checked)
    }

    fn empty_registry() -> ContextTypeRegistry {
        ContextTypeRegistry::all_system(FxHashMap::default())
    }

    #[test]
    fn integration_text_only() {
        let i = Interner::new();
        let result = compile_src(&i, "hello world", &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_emit() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            r#"{{ "hello" }}"#,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_context_read_var_write() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            "{{ $count = 42 }}",
            &empty_registry(),
        );
        assert!(result.is_ok());

        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("count"), Ty::Int)]),
        );
        let result = compile_src(&i, "{{ @count | to_string }}", &reg);
        assert!(result.is_ok());
    }

    #[test]
    fn integration_match_with_catch_all() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("name"), Ty::String)]),
        );
        let result = compile_src(
            &i,
            r#"{{ x = @name }}{{ x }}{{_}}default{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_variable_binding() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("name"), Ty::String)]),
        );
        let result = compile_src(&i, r#"{{ x = @name }}{{ x }}"#, &reg);
        assert!(result.is_ok());
    }

    #[test]
    fn integration_extern_fn() {
        let i = Interner::new();
        // Extern functions are now provided via context_types with Ty::Fn { kind: FnKind::Extern, .. }
        let reg = ContextTypeRegistry::all_system(FxHashMap::from_iter([(
            i.intern("fetch_user"),
            Ty::Fn {
                params: vec![Ty::Int],
                ret: Box::new(Ty::String),
                kind: FnKind::Extern,
                captures: vec![],
                effect: Effect::Pure,
            },
        )]));
        let result = compile_src(
            &i,
            r#"{{ x = @fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_pipe_with_lambda() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]),
        );
        let result = compile_src(
            &i,
            r#"{{ x = @items | filter(x -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_field_access() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(FxHashMap::from_iter([(
            i.intern("user"),
            Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])),
        )]));
        let result = compile_src(&i, "{{ @user.name }}", &reg);
        assert!(result.is_ok());
    }

    #[test]
    fn integration_nested_match() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(FxHashMap::from_iter([(
            i.intern("users"),
            Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])))),
        )]));
        let result = compile_src(
            &i,
            r#"{{ { name, } = @users }}{{ name }}{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_type_error_int_emit() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ 42 }}", &empty_registry());
        assert!(result.is_err());
    }

    #[test]
    fn integration_range_expression() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            "{{ x in 0..10 }}{{ x | to_string }}{{/}}",
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_object_pattern() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(FxHashMap::from_iter([(
            i.intern("data"),
            Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("value"), Ty::Int),
            ])),
        )]));
        let result = compile_src(
            &i,
            r#"{{ { name, } = @data }}{{ name }}{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_multi_arm() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("role"), Ty::String)]),
        );
        let result = compile_src(
            &i,
            r#"{{ "admin" = @role }}admin page{{ "user" }}user page{{_}}guest{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_list_destructure() {
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]),
        );
        let result = compile_src(
            &i,
            r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}"#,
            &reg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_string_concat() {
        let i = Interner::new();
        let result = compile_src(
            &i,
            r#"{{ "hello" + " " + "world" }}"#,
            &empty_registry(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn integration_boolean_logic() {
        let i = Interner::new();
        let result = compile_src(&i, "{{ true }}", &empty_registry());
        assert!(result.is_err());
    }

    fn compile_script_src(
        interner: &Interner,
        script: &acvus_ast::Script,
        registry: &ContextTypeRegistry,
    ) -> Result<(MirModule, HintTable, Ty), Vec<MirError>> {
        let mut subst = TySubst::new();
        let unchecked = typecheck_script(interner, script, registry, None, &mut subst)?;
        let checked = check_completeness(unchecked, &subst)?;
        let tail_ty = checked.tail_ty.clone();
        let (module, hints) = lower_checked_script(interner, script, checked)?;
        Ok((module, hints, tail_ty))
    }

    fn compile_script_test(source: &str) -> MirModule {
        let i = Interner::new();
        let script = acvus_ast::parse_script(&i, source).unwrap();
        let registry = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("data"), Ty::String)]),
        );
        let mut subst = TySubst::new();
        let unchecked = typecheck_script(&i, &script, &registry, None, &mut subst).unwrap();
        let checked = check_completeness(unchecked, &subst).unwrap();
        let (module, _) = lower_checked_script(&i, &script, checked).unwrap();
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

    // ── Non-pure context load tests ──────────────────────────────────

    fn extern_fn_context() -> (Interner, ContextTypeRegistry) {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([
            (
                i.intern("mapper"),
                Ty::Fn {
                    params: vec![Ty::Int],
                    ret: Box::new(Ty::String),
                    kind: FnKind::Extern,
                    captures: vec![],
                    effect: Effect::Pure,
                },
            ),
            (i.intern("items"), Ty::List(Box::new(Ty::Int))),
        ]);
        (i, ContextTypeRegistry::all_system(ctx))
    }

    /// Pipe with extern fn on the right: `@items | @mapper` should work.
    /// The pipe desugars to @mapper(@items) — call position allows non-pure.
    #[test]
    fn pipe_extern_fn_ok() {
        let (i, reg) = extern_fn_context();
        let result = compile_src(
            &i,
            r#"{{ x = @items | map(i -> @mapper(i)) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
            &reg,
        );
        assert!(result.is_ok(), "pipe with extern fn call should work: {result:?}");
    }

    /// Direct call: `@mapper(42)` should work.
    #[test]
    fn direct_extern_fn_call_ok() {
        let (i, reg) = extern_fn_context();
        let result = compile_src(
            &i,
            "{{ @mapper(42) }}",
            &reg,
        );
        assert!(result.is_ok(), "direct extern fn call should work: {result:?}");
    }

    /// Bare load: `@mapper` without calling is now allowed (Fn is Lazy, not Unpure).
    #[test]
    fn bare_extern_fn_load_ok() {
        let (i, reg) = extern_fn_context();
        let result = compile_src(
            &i,
            "{{ x = @mapper }}{{ x(1) }}{{_}}{{/}}",
            &reg,
        );
        assert!(result.is_ok(), "bare extern fn load should succeed (Lazy tier): {result:?}");
    }

    /// Script: `@mapper(1)` should work.
    #[test]
    fn script_extern_fn_call_ok() {
        let (i, reg) = extern_fn_context();
        let script = acvus_ast::parse_script(&i, "@mapper(1)").unwrap();
        let result = compile_script_src(&i, &script, &reg);
        assert!(result.is_ok(), "script extern fn call should work: {result:?}");
    }

    /// Script: bare `@mapper` as tail is now allowed (Fn is Lazy, not Unpure).
    #[test]
    fn script_bare_extern_fn_ok() {
        let (i, reg) = extern_fn_context();
        let script = acvus_ast::parse_script(&i, "@mapper").unwrap();
        let result = compile_script_src(&i, &script, &reg);
        assert!(result.is_ok(), "script bare extern fn should succeed (Lazy tier): {result:?}");
    }

    /// Script pipe: `@items | @mapper` should work (pipe = call position).
    #[test]
    fn script_pipe_extern_fn_ok() {
        let (i, _) = extern_fn_context();
        let ctx = FxHashMap::from_iter([
            (
                i.intern("mapper"),
                Ty::Fn {
                    params: vec![Ty::List(Box::new(Ty::Int))],
                    ret: Box::new(Ty::String),
                    kind: FnKind::Extern,
                    captures: vec![],
                    effect: Effect::Pure,
                },
            ),
            (i.intern("items"), Ty::List(Box::new(Ty::Int))),
        ]);
        let reg = ContextTypeRegistry::all_system(ctx);
        let script = acvus_ast::parse_script(&i, "@items | @mapper").unwrap();
        let result = compile_script_src(&i, &script, &reg);
        assert!(result.is_ok(), "script pipe extern fn should work: {result:?}");
    }

    // ── Layered API tests ───────────────────────────────────────────

    #[test]
    fn layer_typecheck_produces_unchecked() {
        let i = Interner::new();
        let reg = empty_registry();
        let script = acvus_ast::parse_script(&i, "1 + 2").unwrap();
        let mut subst = TySubst::new();
        let unchecked = typecheck_script(&i, &script, &reg, None, &mut subst);
        assert!(unchecked.is_ok());
        let unchecked = unchecked.unwrap();
        assert_eq!(unchecked.tail_ty, Ty::Int);
    }

    #[test]
    fn layer_check_completeness_promotes() {
        let i = Interner::new();
        let reg = empty_registry();
        let script = acvus_ast::parse_script(&i, "1 + 2").unwrap();
        let mut subst = TySubst::new();
        let unchecked = typecheck_script(&i, &script, &reg, None, &mut subst).unwrap();
        let checked = check_completeness(unchecked, &subst);
        assert!(checked.is_ok());
    }

    #[test]
    fn layer_lower_checked_succeeds() {
        let i = Interner::new();
        let reg = empty_registry();
        let script = acvus_ast::parse_script(&i, "1 + 2").unwrap();
        let mut subst = TySubst::new();
        let unchecked = typecheck_script(&i, &script, &reg, None, &mut subst).unwrap();
        let checked = check_completeness(unchecked, &subst).unwrap();
        let result = lower_checked_script(&i, &script, checked);
        assert!(result.is_ok());
    }

    #[test]
    fn layer_shared_subst_resolves_across_units() {
        // SCC scenario: two scripts share a TySubst.
        // Script 1: `[]` — element type unknown (Unchecked has unresolved var)
        // Script 2: `@data` — resolves to String
        // After both typecheck, the shared subst has enough info.
        let i = Interner::new();
        let reg = ContextTypeRegistry::all_system(
            FxHashMap::from_iter([(i.intern("data"), Ty::String)]),
        );
        let mut subst = TySubst::new();

        // Script 1: `[]` with hint = List<Var> (element type unknown)
        let hint_elem = subst.fresh_var();
        let hint = Ty::List(Box::new(hint_elem.clone()));
        let script1 = acvus_ast::parse_script(&i, "[]").unwrap();
        let unchecked1 = typecheck_script(&i, &script1, &reg, Some(&hint), &mut subst).unwrap();

        // At this point, element type is unresolved.
        assert!(
            crate::typeck::contains_var(&subst.resolve(&unchecked1.tail_ty)),
            "element type should be unresolved after script 1 alone"
        );

        // Script 2: `@data` — resolves hint_elem to String via unification
        let script2 = acvus_ast::parse_script(&i, "@data").unwrap();
        let _unchecked2 = typecheck_script(&i, &script2, &reg, Some(&hint_elem), &mut subst).unwrap();

        // Now check_completeness on script 1 should succeed — var resolved by script 2.
        let checked1 = check_completeness(unchecked1, &subst);
        assert!(checked1.is_ok(), "should resolve after shared subst: {checked1:?}");
    }

    #[test]
    fn layer_unresolved_var_rejected_by_completeness() {
        // `[]` without any hint → element type unresolved → check_completeness fails.
        let i = Interner::new();
        let reg = empty_registry();
        let script = acvus_ast::parse_script(&i, "[]").unwrap();
        let mut subst = TySubst::new();
        let unchecked = typecheck_script(&i, &script, &reg, None, &mut subst).unwrap();
        let checked = check_completeness(unchecked, &subst);
        assert!(checked.is_err(), "unresolved var should be rejected");
    }
}
