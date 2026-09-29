pub mod analysis;
pub mod check;
pub mod cfg;
pub mod decl;
pub mod error;
pub mod flows;
pub mod graph;
pub mod infer;
pub mod ir;
pub mod laws;
pub mod lower;
pub mod means;
pub mod step;
pub mod optimize;
pub mod passes;
pub mod printer;
pub mod ser_ty;
pub mod structural;
pub mod ty;
pub mod typeck;
pub mod validate;

#[cfg(test)]
pub(crate) mod test;

#[cfg(test)]
mod tests {
    use crate::ir::{InstKind, MirModule, RefTarget};
    use crate::test::{compile_script, compile_template};
    use crate::ty::{
        Calls, Concrete, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, LenTerm, ObjectTy, Ty, TyKind,
        Types,
    };
    use acvus_utils::{Astr, Interner};
    use rustc_hash::FxHashSet;

    /// A structural object type, its fields sorted by name as
    /// `ObjectTy::Structural` requires.
    fn obj_ty(types: &mut Types<Concrete>, fields: Vec<(Astr, Ty)>) -> Ty {
        let mut fields = fields;
        fields.sort_by_key(|(name, _)| *name);
        types.intern(TyKind::Object(ObjectTy::Structural(fields.into_boxed_slice(), ())))
    }

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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        compile_template(
            &i,
            "% match @n\n% 1 =>\n{{ \"v\" }}\n% _ =>\ndefault\n% end\n",
            &[("n", i64_ty)],
        )
        .unwrap();
    }

    #[test]
    fn a_template_that_moves_a_context_out_is_rejected() {
        let i = Interner::new();
        let mut types = Types::new();
        let age = i.intern("age");
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let user = obj_ty(&mut types, vec![(age, i64_ty)]);
        let err = compile_template(&i, "% let y = @user\n% let z = y.age", &[("user", user)])
            .unwrap_err();
        assert!(err.contains("ContextMovedOut"), "{err}");
    }

    #[test]
    fn integration_variable_binding() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        compile_template(&i, "% let x = @n\n{{ \"v\" }}", &[("n", i64_ty)]).unwrap();
    }

    #[test]
    fn integration_object_field_access() {
        let i = Interner::new();
        let mut types = Types::new();
        let name = i.intern("name");
        let age = i.intern("age");
        let string_ty = types.intern(TyKind::String);
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let user_ty = obj_ty(&mut types, vec![(name, string_ty), (age, i64_ty)]);
        compile_template(&i, "% let x = @user.age", &[("user", user_ty)]).unwrap();
    }

    #[test]
    fn a_string_field_read_out_of_a_context_is_a_copy() {
        let i = Interner::new();
        let mut types = Types::new();
        let name = i.intern("name");
        let age = i.intern("age");
        let string_ty = types.intern(TyKind::String);
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let user_ty = obj_ty(&mut types, vec![(name, string_ty), (age, i64_ty)]);
        compile_template(&i, "{{ @user.name }}", &[("user", user_ty)]).unwrap();
    }

    #[test]
    fn integration_nested_pattern() {
        let i = Interner::new();
        let mut types = Types::new();
        let name = i.intern("name");
        let age = i.intern("age");
        let string_ty = types.intern(TyKind::String);
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let elem_ty = obj_ty(&mut types, vec![(name, string_ty), (age, i64_ty)]);
        let users_ty = types.intern(TyKind::Array(elem_ty, LenTerm::Known(3), ()));
        compile_template(
            &i,
            "% if let [{ name, }, ..] = &@users\n{{ name }}\n% end\n",
            &[("users", users_ty)],
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
        let mut types = Types::new();
        let name = i.intern("name");
        let value = i.intern("value");
        let string_ty = types.intern(TyKind::String);
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let data_ty = obj_ty(&mut types, vec![(name, string_ty), (value, i64_ty)]);
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
        let mut types = Types::new();
        let string_ty = types.intern(TyKind::String);
        compile_template(
            &i,
            "% match @role\n% \"admin\" =>\nadmin page\n% \"user\" =>\nuser page\n% _ =>\nguest\n% end\n",
            &[("role", string_ty)],
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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "@data", &[("data", i64_ty)]).unwrap();
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
        let mut types = Types::new();
        let age = i.intern("age");
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let user = obj_ty(&mut types, vec![(age, i64_ty)]);
        let err = compile_script(&i, "@data", &[("data", user)]).unwrap_err();
        assert!(err.contains("ContextMovedOut"), "{err}");
        let err = compile_script(&i, "let x = @data; x", &[("data", user)]).unwrap_err();
        assert!(err.contains("ContextMovedOut"), "{err}");
    }

    #[test]
    fn a_context_moved_out_and_assigned_back_is_accepted() {
        let i = Interner::new();
        let mut types = Types::new();
        let string_ty = types.intern(TyKind::String);
        compile_script(
            &i,
            r#"let x = @data; @data = "new".to_string(); x"#,
            &[("data", string_ty)],
        )
        .unwrap();
    }

    #[test]
    fn script_bind_and_tail() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "let x = @data; x", &[("data", i64_ty)]).unwrap();
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
        let mut types = Types::new();
        let string_ty = types.intern(TyKind::String);
        let module = compile_script(&i, "let x = @data;", &[("data", string_ty)]).unwrap();
        assert!(module.main.insts.iter().any(|i| matches!(
            &i.kind,
            InstKind::Return { value, .. }
                if module.main.val_types.get(value).map(|ty| module.types.kind(*ty)) == Some(&TyKind::Unit)
        )));
    }

    // -- Extern fn tests ---------------------------------------------

    // -- Context store tests -----------------------------------------

    #[test]
    fn context_store_compiles() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        compile_script(&i, "@x = @x + 1; @x", &[("x", i64_ty)]).unwrap();
    }

    #[test]
    fn context_store_produces_context_store_instruction() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "@x = 42; @x", &[("x", i64_ty)]).unwrap();
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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        compile_script(
            &i,
            "let tmp = @count + 1; @count = tmp; @count",
            &[("count", i64_ty)],
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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "@x", &[("x", i64_ty)]).unwrap();
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
        let mut types = Types::new();
        let age = i.intern("age");
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let obj_ty = obj_ty(&mut types, vec![(age, i64_ty)]);
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
        let mut types = Types::new();
        let val = i.intern("val");
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let obj_ty = obj_ty(&mut types, vec![(val, i64_ty)]);
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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "@x = 42; @x", &[("x", i64_ty)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(kinds.iter().any(|k| matches!(k, InstKind::Commit { .. })));
    }

    #[test]
    fn projection_copy_to_local() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "let x = @data; x", &[("data", i64_ty)]).unwrap();
        let kinds = inst_kinds(&module);
        assert!(kinds.iter().any(|k| matches!(k, InstKind::Return { .. })));
    }

    #[test]
    fn projection_multiple_contexts() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "@a + @b", &[("a", i64_ty), ("b", i64_ty)]).unwrap();
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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let module = compile_script(&i, "@x + 1", &[("x", i64_ty)]).unwrap();
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
        let mut types = Types::new();
        let age = i.intern("age");
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let obj_ty = obj_ty(&mut types, vec![(age, i64_ty)]);
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
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        assert!(compile_script(&i, "@x = 42; @x", &[("x", i64_ty)]).is_ok());
    }

    #[test]
    fn context_data_string() {
        let i = Interner::new();
        let mut types = Types::new();
        let string_ty = types.intern(TyKind::String);
        assert!(compile_script(&i, r#"@x = "hello".to_string(); 1"#, &[("x", string_ty)]).is_ok());
    }

    #[test]
    fn context_data_object() {
        let i = Interner::new();
        let mut types = Types::new();
        let name = i.intern("name");
        let age = i.intern("age");
        let string_ty = types.intern(TyKind::String);
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let obj_ty = obj_ty(&mut types, vec![(name, string_ty), (age, i64_ty)]);
        assert!(compile_script(&i, "@user = @user; 1", &[("user", obj_ty)]).is_ok());
    }

    #[test]
    fn context_data_bool() {
        let i = Interner::new();
        let mut types = Types::new();
        let bool_ty = types.intern(TyKind::Bool);
        assert!(compile_script(&i, "@x = true; @x", &[("x", bool_ty)]).is_ok());
    }

    #[test]
    fn context_fn_rejected() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let fn_ty = types.intern(TyKind::Fn {
            params: Box::new([i64_ty]),
            ret: i64_ty,
            effect: EffectTerm::Known(Effect::opaque()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        // Storing a function to context must fail.
        assert!(compile_script(&i, "@f = @f; @f", &[("f", fn_ty)]).is_err());
    }

    #[test]
    fn context_list_of_fn_rejected() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let fn_ty = types.intern(TyKind::Fn {
            params: Box::new([i64_ty]),
            ret: i64_ty,
            effect: EffectTerm::Known(Effect::opaque()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let list_fn_ty = types.intern(TyKind::Array(fn_ty, LenTerm::Known(3), ()));
        assert!(compile_script(&i, "@x = @x; @x", &[("x", list_fn_ty)]).is_err());
    }

    #[test]
    fn context_object_with_fn_field_rejected() {
        let i = Interner::new();
        let mut types = Types::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let fn_ty = types.intern(TyKind::Fn {
            params: Box::new([i64_ty]),
            ret: i64_ty,
            effect: EffectTerm::Known(Effect::opaque()),
            flows: FlowTerm::Known(Flows::Every),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let name = i.intern("name");
        let callback = i.intern("callback");
        let string_ty = types.intern(TyKind::String);
        let obj_ty = obj_ty(&mut types, vec![(name, string_ty), (callback, fn_ty)]);
        assert!(compile_script(&i, "@obj = @obj; @obj", &[("obj", obj_ty)]).is_err());
    }
}
