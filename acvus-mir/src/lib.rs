pub mod analysis;
pub mod cfg;
mod entry_fetch;
pub mod error;
pub mod flows;
pub mod graph;
pub mod ir;
pub mod laws;
pub mod lower;
pub mod optimize;
pub mod place;
pub mod printer;
mod pvec;
pub mod ser_ty;
pub mod solver;
pub mod structural;
pub mod ty;
pub mod typeck;
pub mod validate;
pub mod variant;

#[cfg(test)]
pub(crate) mod test;

#[cfg(test)]
mod tests {
    use crate::ir::{InstKind, MirModule, RefTarget};
    use crate::test::{compile_script, compile_template};
    use crate::ty::{ObjectTy, Param, Ty};
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
    fn integration_match_with_wildcard_arm() {
        let i = Interner::new();
        compile_template(
            &i,
            "% match @n\n% 1 =>\n{{ \"v\" }}\n% _ =>\ndefault\n% end\n",
            &[("n", Ty::I64)],
        )
        .unwrap();
    }

    #[test]
    fn a_template_that_moves_a_context_out_is_rejected() {
        let i = Interner::new();
        let user = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("age"),
            Ty::I64,
        )])));
        let err = compile_template(&i, "% let y = @user\n% let z = y.age", &[("user", user)])
            .unwrap_err();
        assert!(err.contains("ContextMovedOut"), "{err}");
    }

    #[test]
    fn integration_variable_binding() {
        let i = Interner::new();
        compile_template(&i, "% let x = @n\n{{ \"v\" }}", &[("n", Ty::I64)]).unwrap();
    }

    #[test]
    fn integration_object_field_access() {
        let i = Interner::new();
        let user_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::I64),
        ])));
        compile_template(&i, "% let x = @user.age", &[("user", user_ty)]).unwrap();
    }

    #[test]
    fn a_string_field_read_out_of_a_context_is_a_copy() {
        let i = Interner::new();
        let user_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::I64),
        ])));
        compile_template(&i, "{{ @user.name }}", &[("user", user_ty)]).unwrap();
    }

    #[test]
    fn integration_nested_pattern() {
        let i = Interner::new();
        let users_ty = Ty::Array(
            Box::new(Ty::Object(ObjectTy::written(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::I64),
            ])))),
            crate::ty::LenTerm::Known(3),
        );
        compile_template(
            &i,
            "% if let [{ name, }, ..] = &@users\n{{ name }}\n% end\n",
            &[("users", users_ty.clone())],
        )
        .unwrap();
        let err = compile_template(
            &i,
            "% if let { name, } = @users\n{{ name }}\n% end\n",
            &[("users", users_ty)],
        )
        .unwrap_err();
        assert!(err.contains("incompatible with source type"), "{err}");
    }

    #[test]
    fn integration_type_error_int_emit() {
        let i = Interner::new();
        assert!(compile_template(&i, "{{ 42 }}", &[]).is_err());
    }

    #[test]
    fn integration_object_pattern() {
        let i = Interner::new();
        let data_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("value"), Ty::I64),
        ])));
        compile_template(
            &i,
            "% if let { name, } = @data\n{{ name }}\n% end\n",
            &[("data", data_ty)],
        )
        .unwrap();
    }

    #[test]
    fn integration_match_over_three_arms() {
        let i = Interner::new();
        compile_template(
            &i,
            "% match @role\n% \"admin\" =>\nadmin page\n% \"user\" =>\nuser page\n% _ =>\nguest\n% end\n",
            &[("role", Ty::String)],
        )
        .unwrap();
    }

    #[test]
    fn string_addition_is_a_string_concat() {
        let i = Interner::new();
        let module = compile_template(&i, r#"{{ "hello" + " " + "world" }}"#, &[]).unwrap();
        assert!(
            module
                .main
                .insts
                .iter()
                .any(|i| matches!(&i.kind, InstKind::StringConcat { .. }))
        );
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
        let module = compile_script(&i, "@data", &[("data", Ty::I64)]).unwrap();
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
        let user = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("age"),
            Ty::I64,
        )])));
        let err = compile_script(&i, "@data", &[("data", user.clone())]).unwrap_err();
        assert!(err.contains("ContextMovedOut"), "{err}");
        let err = compile_script(&i, "let x = @data; x", &[("data", user)]).unwrap_err();
        assert!(err.contains("ContextMovedOut"), "{err}");
    }

    #[test]
    fn a_context_moved_out_and_assigned_back_is_accepted() {
        let i = Interner::new();
        compile_script(
            &i,
            r#"let x = @data; @data = "new".to_string(); x"#,
            &[("data", Ty::String)],
        )
        .unwrap();
    }

    #[test]
    fn script_bind_and_tail() {
        let i = Interner::new();
        let module = compile_script(&i, "let x = @data; x", &[("data", Ty::I64)]).unwrap();
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
    fn script_trailing_semicolon_returns_unit() {
        let i = Interner::new();
        let module = compile_script(&i, "let x = @data;", &[("data", Ty::String)]).unwrap();
        assert!(module.main.insts.iter().any(|i| matches!(
            &i.kind,
            InstKind::Return { value, .. } if module.main.val_types.get(value) == Some(&Ty::Unit)
        )));
    }

    // -- Extern fn tests ---------------------------------------------

    // -- Context store tests -----------------------------------------

    #[test]
    fn context_store_compiles() {
        let i = Interner::new();
        compile_script(&i, "@x = @x + 1; @x", &[("x", Ty::I64)]).unwrap();
    }

    #[test]
    fn context_store_produces_context_store_instruction() {
        let i = Interner::new();
        let module = compile_script(&i, "@x = 42; @x", &[("x", Ty::I64)]).unwrap();
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
            "let tmp = @count + 1; @count = tmp; @count",
            &[("count", Ty::I64)],
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
        let module = compile_script(&i, "@x", &[("x", Ty::I64)]).unwrap();
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
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("age"),
            Ty::I64,
        )])));
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
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("val"),
            Ty::I64,
        )])));
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
        let module = compile_script(&i, "@x = 42; @x", &[("x", Ty::I64)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(kinds.iter().any(|k| matches!(k, InstKind::Commit { .. })));
    }

    #[test]
    fn projection_copy_to_local() {
        let i = Interner::new();
        let module = compile_script(&i, "let x = @data; x", &[("data", Ty::I64)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(kinds.iter().any(|k| matches!(k, InstKind::Return { .. })));
    }

    #[test]
    fn projection_multiple_contexts() {
        let i = Interner::new();
        let module = compile_script(&i, "@a + @b", &[("a", Ty::I64), ("b", Ty::I64)]).unwrap();
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
        let module = compile_script(&i, "@x + 1", &[("x", Ty::I64)]).unwrap();
        let mut ref_dsts = FxHashSet::default();
        let mut consumed = FxHashSet::default();
        for inst in &module.main.insts {
            match &inst.kind {
                InstKind::Ref { dst, .. } => {
                    ref_dsts.insert(*dst);
                }
                InstKind::Take {
                    target: RefTarget::Through(r),
                    ..
                }
                | InstKind::Assign {
                    target: RefTarget::Through(r),
                    ..
                } => {
                    consumed.insert(*r);
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
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("age"),
            Ty::I64,
        )])));
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
                InstKind::Take {
                    target: RefTarget::Through(r),
                    ..
                }
                | InstKind::Assign {
                    target: RefTarget::Through(r),
                    ..
                } => {
                    consumed.insert(*r);
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
        assert!(compile_script(&i, "@x = 42; @x", &[("x", Ty::I64)]).is_ok());
    }

    #[test]
    fn context_data_string() {
        let i = Interner::new();
        assert!(compile_script(&i, r#"@x = "hello".to_string(); 1"#, &[("x", Ty::String)]).is_ok());
    }

    #[test]
    fn context_data_object() {
        let i = Interner::new();
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::I64),
        ])));
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
            params: vec![Param::new(i.intern("x"), Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![],

            effect: crate::ty::Effect::OPAQUE.into(),
            flows: crate::ty::Flows::Every.into(),
        };
        // Storing a function to context must fail.
        assert!(compile_script(&i, "@f = @f; @f", &[("f", fn_ty)]).is_err());
    }

    #[test]
    fn context_list_of_fn_rejected() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![],

            effect: crate::ty::Effect::OPAQUE.into(),
            flows: crate::ty::Flows::Every.into(),
        };
        let list_fn_ty = Ty::Array(Box::new(fn_ty), crate::ty::LenTerm::Known(3));
        assert!(compile_script(&i, "@x = @x; @x", &[("x", list_fn_ty)]).is_err());
    }

    #[test]
    fn context_object_with_fn_field_rejected() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![Param::new(i.intern("x"), Ty::I64)],
            ret: Box::new(Ty::I64),
            captures: vec![],

            effect: crate::ty::Effect::OPAQUE.into(),
            flows: crate::ty::Flows::Every.into(),
        };
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("callback"), fn_ty),
        ])));
        assert!(compile_script(&i, "@obj = @obj; @obj", &[("obj", obj_ty)]).is_err());
    }
}
