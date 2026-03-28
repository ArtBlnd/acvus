pub mod analysis;
pub mod builtins;
pub mod error;
pub mod graph;
pub mod hints;
pub mod ir;
pub mod lower;
pub mod optimize;
pub mod pass;
pub mod printer;
pub mod ser_ty;
pub mod ssa;
pub mod ssa_pass;
pub mod ty;
pub mod typeck;
pub mod validate;
pub mod variant;

#[cfg(test)]
pub(crate) mod test;

pub use pass::AnalysisPass;

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::graph::QualifiedRef;
use crate::ty::Ty;

/// Build a QualifiedRef→Ty mapping from a simple name→Ty context map.
pub fn build_context_ids(
    context: &FxHashMap<Astr, Ty>,
) -> acvus_utils::Freeze<FxHashMap<QualifiedRef, Ty>> {
    acvus_utils::Freeze::new(
        context
            .iter()
            .map(|(&name, ty)| (QualifiedRef::root(name), ty.clone()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use crate::ir::{InstKind, MirModule};
    use crate::test::{compile_script, compile_template};
    use crate::ty::{Effect, Param, Ty};
    use acvus_utils::Interner;
    use rustc_hash::{FxHashMap, FxHashSet};

    // ── Template integration tests ──────────────────────────────────

    #[test]
    fn integration_text_only() {
        let i = Interner::new();
        compile_template(&i, "hello world", &[]).unwrap();
    }

    #[test]
    fn integration_string_emit() {
        let i = Interner::new();
        compile_template(&i, r#"{{ "hello" }}"#, &[]).unwrap();
    }

    #[test]
    fn integration_match_with_catch_all() {
        let i = Interner::new();
        compile_template(
            &i,
            r#"{{ x = @name }}{{ x }}{{_}}default{{/}}"#,
            &[("name", Ty::String)],
        )
        .unwrap();
    }

    #[test]
    fn integration_variable_binding() {
        let i = Interner::new();
        compile_template(&i, r#"{{ x = @name }}{{ x }}"#, &[("name", Ty::String)]).unwrap();
    }

    #[test]
    fn integration_extern_fn() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("_"), Ty::Int)],
            ret: Box::new(Ty::String),
            captures: vec![],
            effect: Effect::pure(),
        };
        compile_template(
            &i,
            r#"{{ x = @fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            &[("fetch_user", fn_ty)],
        )
        .unwrap();
    }

    #[test]
    fn integration_object_field_access() {
        let i = Interner::new();
        let user_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        compile_template(&i, "{{ @user.name }}", &[("user", user_ty)]).unwrap();
    }

    #[test]
    fn integration_nested_match() {
        let i = Interner::new();
        let users_ty = Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]))));
        compile_template(
            &i,
            r#"{{ { name, } = @users }}{{ name }}{{/}}"#,
            &[("users", users_ty)],
        )
        .unwrap();
    }

    #[test]
    fn integration_type_error_int_emit() {
        let i = Interner::new();
        assert!(compile_template(&i, "{{ 42 }}", &[]).is_err());
    }

    #[test]
    fn integration_object_pattern() {
        let i = Interner::new();
        let data_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("value"), Ty::Int),
        ]));
        compile_template(
            &i,
            r#"{{ { name, } = @data }}{{ name }}{{/}}"#,
            &[("data", data_ty)],
        )
        .unwrap();
    }

    #[test]
    fn integration_multi_arm() {
        let i = Interner::new();
        compile_template(
            &i,
            r#"{{ "admin" = @role }}admin page{{ "user" }}user page{{_}}guest{{/}}"#,
            &[("role", Ty::String)],
        )
        .unwrap();
    }

    #[test]
    fn integration_string_concat() {
        let i = Interner::new();
        compile_template(&i, r#"{{ "hello" + " " + "world" }}"#, &[]).unwrap();
    }

    #[test]
    fn integration_boolean_logic() {
        let i = Interner::new();
        assert!(compile_template(&i, "{{ true }}", &[]).is_err());
    }

    // ── Script tests ────────────────────────────────────────────────

    #[test]
    fn script_single_expr() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "@data", &[("data", Ty::String)]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Return(_)))
        );
    }

    #[test]
    fn script_bind_and_tail() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "x = @data; x", &[("data", Ty::String)]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::ContextLoad { .. }))
        );
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Return(_)))
        );
    }

    #[test]
    fn script_trailing_semicolon_no_yield() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "x = @data;", &[("data", Ty::String)]).unwrap();
        assert!(
            !module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Return(_)))
        );
    }

    // ── Extern fn tests ─────────────────────────────────────────────

    fn extern_fn_ctx(i: &Interner) -> Vec<(&'static str, Ty)> {
        vec![
            (
                "mapper",
                Ty::Fn {
                    params: vec![Param::new(i.intern("_"), Ty::Int)],
                    ret: Box::new(Ty::String),

                    captures: vec![],
                    effect: Effect::pure(),
                },
            ),
            ("items", Ty::List(Box::new(Ty::Int))),
        ]
    }

    #[test]
    fn direct_extern_fn_call_ok() {
        let i = Interner::new();
        let ctx = extern_fn_ctx(&i);
        compile_template(&i, "{{ @mapper(42) }}", &ctx).unwrap();
    }

    #[test]
    fn bare_extern_fn_load_ok() {
        let i = Interner::new();
        let ctx = extern_fn_ctx(&i);
        compile_template(&i, "{{ x = @mapper }}{{ x(1) }}{{_}}{{/}}", &ctx).unwrap();
    }

    #[test]
    fn script_extern_fn_call_ok() {
        let i = Interner::new();
        let ctx = extern_fn_ctx(&i);
        compile_script(&i, "@mapper(1)", &ctx).unwrap();
    }

    #[test]
    fn script_bare_extern_fn_ok() {
        let i = Interner::new();
        let ctx = extern_fn_ctx(&i);
        compile_script(&i, "@mapper", &ctx).unwrap();
    }

    #[test]
    fn script_pipe_extern_fn_ok() {
        let i = Interner::new();
        let ctx = vec![
            (
                "mapper",
                Ty::Fn {
                    params: vec![Param::new(i.intern("_"), Ty::List(Box::new(Ty::Int)))],
                    ret: Box::new(Ty::String),
                    captures: vec![],
                    effect: Effect::pure(),
                },
            ),
            ("items", Ty::List(Box::new(Ty::Int))),
        ];
        compile_script(&i, "@items | @mapper", &ctx).unwrap();
    }

    // ── Context store tests ─────────────────────────────────────────

    #[test]
    fn context_store_compiles() {
        let i = Interner::new();
        compile_script(&i, "@x = @x + 1; @x", &[("x", Ty::Int)]).unwrap();
    }

    #[test]
    fn context_store_produces_context_store_instruction() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|inst| matches!(&inst.kind, InstKind::ContextStore { .. }))
        );
    }

    #[test]
    fn context_store_roundtrip() {
        let i = Interner::new();
        compile_script(
            &i,
            "tmp = @count + 1; @count = tmp; @count",
            &[("count", Ty::Int)],
        )
        .unwrap();
    }

    // ── Projection IR structure tests ───────────────────────────────

    fn inst_kinds(module: &MirModule) -> Vec<&InstKind> {
        module.main.insts.iter().map(|i| &i.kind).collect()
    }

    #[test]
    fn projection_bare_context_read() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "@x", &[("x", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        let proj_idx = kinds
            .iter()
            .position(|k| matches!(k, InstKind::ContextProject { .. }));
        let load_idx = kinds
            .iter()
            .position(|k| matches!(k, InstKind::ContextLoad { .. }));
        assert!(proj_idx.is_some() && load_idx.is_some());
        assert!(proj_idx.unwrap() < load_idx.unwrap());
    }

    #[test]
    fn projection_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let (module, _) = compile_script(&i, "@obj.name", &[("obj", obj_ty)]).unwrap();
        let kinds = inst_kinds(&module);
        // SSA pass: ContextProject → ContextLoad (materialize object) → FieldGet (on value).
        let proj = kinds
            .iter()
            .position(|k| matches!(k, InstKind::ContextProject { .. }))
            .unwrap();
        let load = kinds
            .iter()
            .position(|k| matches!(k, InstKind::ContextLoad { .. }))
            .unwrap();
        let field = kinds
            .iter()
            .position(|k| matches!(k, InstKind::FieldGet { .. }))
            .unwrap();
        assert!(proj < load && load < field);
    }

    #[test]
    fn projection_loaded_before_binop() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("val"), Ty::Int)]));
        let (module, _) = compile_script(&i, "@obj.val + 1", &[("obj", obj_ty)]).unwrap();
        let kinds = inst_kinds(&module);
        let load = kinds
            .iter()
            .position(|k| matches!(k, InstKind::ContextLoad { .. }))
            .unwrap();
        let binop = kinds
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { .. }))
            .unwrap();
        assert!(load < binop);
    }

    #[test]
    fn projection_context_store() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        let store_i = kinds
            .iter()
            .position(|k| matches!(k, InstKind::ContextStore { .. }))
            .unwrap();
        assert!(store_i > 0 && matches!(kinds[store_i - 1], InstKind::ContextProject { .. }));
    }

    #[test]
    fn projection_copy_to_local() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "x = @data; x", &[("data", Ty::String)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, InstKind::ContextProject { .. }))
        );
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, InstKind::ContextLoad { .. }))
        );
    }

    #[test]
    fn projection_multiple_contexts() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "@a + @b", &[("a", Ty::Int), ("b", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        assert_eq!(
            kinds
                .iter()
                .filter(|k| matches!(k, InstKind::ContextProject { .. }))
                .count(),
            2
        );
        assert_eq!(
            kinds
                .iter()
                .filter(|k| matches!(k, InstKind::ContextLoad { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn projection_no_leak_simple() {
        let i = Interner::new();
        let (module, _) = compile_script(&i, "@x + 1", &[("x", Ty::Int)]).unwrap();
        let mut proj_dsts = FxHashSet::default();
        let mut consumed = FxHashSet::default();
        for inst in &module.main.insts {
            match &inst.kind {
                InstKind::ContextProject { dst, .. } => {
                    proj_dsts.insert(*dst);
                }
                InstKind::ContextLoad { src, .. } => {
                    consumed.insert(*src);
                }
                InstKind::ContextStore { dst, .. } => {
                    consumed.insert(*dst);
                }
                InstKind::FieldGet { object, .. } => {
                    consumed.insert(*object);
                }
                _ => {}
            }
        }
        for dst in &proj_dsts {
            assert!(consumed.contains(dst), "projection leaked");
        }
    }

    #[test]
    fn projection_no_leak_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let (module, _) = compile_script(&i, "@obj.name", &[("obj", obj_ty)]).unwrap();
        let mut proj_dsts = FxHashSet::default();
        let mut consumed = FxHashSet::default();
        for inst in &module.main.insts {
            match &inst.kind {
                InstKind::ContextProject { dst, .. } => {
                    proj_dsts.insert(*dst);
                }
                InstKind::FieldGet { object, .. } => {
                    consumed.insert(*object);
                }
                InstKind::ContextLoad { src, .. } => {
                    consumed.insert(*src);
                }
                InstKind::ContextStore { dst, .. } => {
                    consumed.insert(*dst);
                }
                _ => {}
            }
        }
        for dst in &proj_dsts {
            assert!(consumed.contains(dst), "projection leaked");
        }
    }

    // ── Materiality: context store validation ─────────────────────────
    //
    // Soundness: non-materializable types must be rejected.
    // Completeness: materializable types must be accepted.

    // ── Completeness: materializable types accepted ──

    #[test]
    fn materiality_store_int() {
        let i = Interner::new();
        assert!(compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).is_ok());
    }

    #[test]
    fn materiality_store_string() {
        let i = Interner::new();
        assert!(compile_script(&i, r#"@x = "hello"; @x"#, &[("x", Ty::String)]).is_ok());
    }

    #[test]
    fn materiality_store_deque() {
        let i = Interner::new();
        let o = crate::ty::TySubst::new().alloc_identity(false);
        assert!(
            compile_script(
                &i,
                "@x = [1, 2, 3]; @x",
                &[("x", Ty::Deque(Box::new(Ty::Int), Box::new(o)))]
            )
            .is_ok()
        );
    }

    #[test]
    fn materiality_store_object() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        assert!(compile_script(&i, "@user = @user; @user", &[("user", obj_ty)]).is_ok());
    }

    #[test]
    fn materiality_store_bool() {
        let i = Interner::new();
        assert!(compile_script(&i, "@x = true; @x", &[("x", Ty::Bool)]).is_ok());
    }

    // ── Soundness: non-materializable types rejected ──

    #[test]
    fn materiality_reject_fn_in_context() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![],
            effect: Effect::pure(),
        };
        // Storing a function to context must fail.
        assert!(compile_script(&i, "@f = @f; @f", &[("f", fn_ty)]).is_err());
    }

    // materiality_reject_iterator_in_context: migrated to acvus-mir-test
    // (Iterator is now UserDefined, requires TypeRegistry + Interner).

    // ── Soundness: nested non-materializable rejected ──

    #[test]
    fn materiality_reject_list_of_fn() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![],
            effect: Effect::pure(),
        };
        let list_fn_ty = Ty::List(Box::new(fn_ty));
        assert!(compile_script(&i, "@x = @x; @x", &[("x", list_fn_ty)]).is_err());
    }

    #[test]
    fn materiality_reject_object_with_fn_field() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![],
            effect: Effect::pure(),
        };
        let obj_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("callback"), fn_ty),
        ]));
        assert!(compile_script(&i, "@obj = @obj; @obj", &[("obj", obj_ty)]).is_err());
    }
}
