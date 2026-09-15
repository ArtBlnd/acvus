pub mod analysis;
pub mod cfg;
pub mod error;
pub mod graph;
pub mod ir;
pub mod lower;
pub mod optimize;
pub mod printer;
pub mod ser_ty;
pub mod solver;
pub mod ty;
pub mod typeck;
pub mod validate;
pub mod variant;

#[cfg(test)]
pub(crate) mod test;

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::graph::QualifiedRef;
use crate::ty::Ty;

/// Build a QualifiedRef->Ty mapping from a simple name->Ty context map.
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
    use crate::ty::{Param, Ty};
    use acvus_utils::Interner;
    use rustc_hash::{FxHashMap, FxHashSet};

    // -- Template integration tests ----------------------------------

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
    fn integration_object_field_access() {
        let i = Interner::new();
        let user_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        compile_template(&i, "{{ x = @user.age }}", &[("user", user_ty)]).unwrap();
    }

    #[test]
    fn a_string_field_read_out_of_a_context_is_rejected() {
        let i = Interner::new();
        let user_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        let err = compile_template(&i, "{{ @user.name }}", &[("user", user_ty)]).unwrap_err();
        assert!(err.contains("UseAfterMove"), "{err}");
    }

    #[test]
    fn integration_nested_match() {
        let i = Interner::new();
        let users_ty = Ty::Array(
            Box::new(Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ]))),
            crate::ty::LenTerm::Known(3),
        );
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

    // -- Script tests ------------------------------------------------

    #[test]
    fn script_single_expr() {
        let i = Interner::new();
        let module = compile_script(&i, "@data", &[("data", Ty::Int)]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Return { .. }))
        );
    }

    #[test]
    fn a_context_moved_out_and_not_assigned_back_is_rejected() {
        let i = Interner::new();
        let err = compile_script(&i, "@data", &[("data", Ty::String)]).unwrap_err();
        assert!(err.contains("UseAfterMove"), "{err}");
        let err = compile_script(&i, "x = @data; x", &[("data", Ty::String)]).unwrap_err();
        assert!(err.contains("UseAfterMove"), "{err}");
    }

    #[test]
    fn a_context_moved_out_and_assigned_back_is_accepted() {
        let i = Interner::new();
        compile_script(&i, r#"x = @data; @data = "new"; x"#, &[("data", Ty::String)]).unwrap();
    }

    #[test]
    fn script_bind_and_tail() {
        let i = Interner::new();
        let module = compile_script(&i, "x = @data; x", &[("data", Ty::Int)]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Fetch { .. }))
        );
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Return { .. }))
        );
    }

    #[test]
    fn script_trailing_semicolon_no_yield() {
        let i = Interner::new();
        let module = compile_script(&i, "x = @data;", &[("data", Ty::String)]).unwrap();
        assert!(
            !module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Return { .. }))
        );
    }

    // -- Extern fn tests ---------------------------------------------


    // -- Context store tests -----------------------------------------

    #[test]
    fn context_store_compiles() {
        let i = Interner::new();
        compile_script(&i, "@x = @x + 1; @x", &[("x", Ty::Int)]).unwrap();
    }

    #[test]
    fn context_store_produces_context_store_instruction() {
        let i = Interner::new();
        let module = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|inst| matches!(&inst.kind, InstKind::Commit { .. }))
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

    // -- Projection IR structure tests -------------------------------

    fn inst_kinds(module: &MirModule) -> Vec<&InstKind> {
        module.main.insts.iter().map(|i| &i.kind).collect()
    }

    #[test]
    fn projection_bare_context_read() {
        let i = Interner::new();
        let module = compile_script(&i, "@x", &[("x", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        let take_idx = kinds
            .iter()
            .position(|k| matches!(k, InstKind::Fetch { .. }));
        let ret_idx = kinds
            .iter()
            .position(|k| matches!(k, InstKind::Return { .. }));
        assert!(take_idx.is_some() && ret_idx.is_some());
        assert!(take_idx.unwrap() < ret_idx.unwrap());
    }

    #[test]
    fn projection_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::Int)]));
        let module = compile_script(&i, "@obj.age", &[("obj", obj_ty)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(
            kinds.iter().any(|k| matches!(k, InstKind::Return { .. })),
            "should compile and return"
        );
    }

    #[test]
    fn projection_loaded_before_binop() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("val"), Ty::Int)]));
        let module = compile_script(&i, "@obj.val + 1", &[("obj", obj_ty)]).unwrap();
        let kinds = inst_kinds(&module);
        let take = kinds
            .iter()
            .position(|k| matches!(k, InstKind::Take { .. }))
            .unwrap();
        let binop = kinds
            .iter()
            .position(|k| matches!(k, InstKind::BinOp { .. }))
            .unwrap();
        assert!(take < binop);
    }

    #[test]
    fn projection_context_store() {
        let i = Interner::new();
        let module = compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(kinds.iter().any(|k| matches!(k, InstKind::Commit { .. })));
    }

    #[test]
    fn projection_copy_to_local() {
        let i = Interner::new();
        let module = compile_script(&i, "x = @data; x", &[("data", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(kinds.iter().any(|k| matches!(k, InstKind::Return { .. })));
    }

    #[test]
    fn projection_multiple_contexts() {
        let i = Interner::new();
        let module = compile_script(&i, "@a + @b", &[("a", Ty::Int), ("b", Ty::Int)]).unwrap();
        let kinds = inst_kinds(&module);
        assert_eq!(
            kinds
                .iter()
                .filter(|k| matches!(k, InstKind::Fetch { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn projection_no_leak_simple() {
        let i = Interner::new();
        let module = compile_script(&i, "@x + 1", &[("x", Ty::Int)]).unwrap();
        let mut ref_dsts = FxHashSet::default();
        let mut consumed = FxHashSet::default();
        for inst in &module.main.insts {
            match &inst.kind {
                InstKind::Ref { dst, .. } => {
                    ref_dsts.insert(*dst);
                }
                InstKind::Load { src, .. } => {
                    consumed.insert(*src);
                }
                InstKind::Store { dst, .. } => {
                    consumed.insert(*dst);
                }
                InstKind::FieldGet { object, .. } => {
                    consumed.insert(*object);
                }
                _ => {}
            }
        }
        for dst in &ref_dsts {
            assert!(consumed.contains(dst), "projection leaked");
        }
    }

    #[test]
    fn projection_no_leak_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("age"), Ty::Int)]));
        let module = compile_script(&i, "@obj.age", &[("obj", obj_ty)]).unwrap();
        let mut ref_dsts = FxHashSet::default();
        let mut consumed = FxHashSet::default();
        for inst in &module.main.insts {
            match &inst.kind {
                InstKind::Ref { dst, .. } => {
                    ref_dsts.insert(*dst);
                }
                InstKind::FieldGet { object, .. } => {
                    consumed.insert(*object);
                }
                InstKind::Load { src, .. } => {
                    consumed.insert(*src);
                }
                InstKind::Store { dst, .. } => {
                    consumed.insert(*dst);
                }
                _ => {}
            }
        }
        for dst in &ref_dsts {
            assert!(consumed.contains(dst), "projection leaked");
        }
    }

    // -- A context holds data (RFC-0014) ---------------------------------

    #[test]
    fn context_data_int() {
        let i = Interner::new();
        assert!(compile_script(&i, "@x = 42; @x", &[("x", Ty::Int)]).is_ok());
    }

    #[test]
    fn context_data_string() {
        let i = Interner::new();
        assert!(compile_script(&i, r#"@x = "hello"; 1"#, &[("x", Ty::String)]).is_ok());
    }

    #[test]
    fn context_data_object() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        assert!(compile_script(&i, "@user = @user; 1", &[("user", obj_ty)]).is_ok());
    }

    #[test]
    fn context_data_bool() {
        let i = Interner::new();
        assert!(compile_script(&i, "@x = true; @x", &[("x", Ty::Bool)]).is_ok());
    }


    #[test]
    fn context_fn_rejected() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![],

            effect: crate::ty::Effect::OPAQUE.into(),
        };
        // Storing a function to context must fail.
        assert!(compile_script(&i, "@f = @f; @f", &[("f", fn_ty)]).is_err());
    }


    #[test]
    fn context_list_of_fn_rejected() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![],

            effect: crate::ty::Effect::OPAQUE.into(),
        };
        let list_fn_ty = Ty::Array(Box::new(fn_ty), crate::ty::LenTerm::Known(3));
        assert!(compile_script(&i, "@x = @x; @x", &[("x", list_fn_ty)]).is_err());
    }

    #[test]
    fn context_object_with_fn_field_rejected() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::Int)],
            ret: Box::new(Ty::Int),
            captures: vec![],

            effect: crate::ty::Effect::OPAQUE.into(),
        };
        let obj_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("callback"), fn_ty),
        ]));
        assert!(compile_script(&i, "@obj = @obj; @obj", &[("obj", obj_ty)]).is_err());
    }
}
