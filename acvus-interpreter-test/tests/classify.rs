//! A context read through references while temporaries are built from it (RFC-0018).

use acvus_extern::Owned;
use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::{ObjectTy, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[tokio::test]
async fn a_context_string_rebuilt_from_itself_through_temporaries() {
    let i = Interner::new();
    let name = i.intern("name");
    let age = i.intern("age");
    let user = typed(
        Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (name, Ty::String),
            (age, Ty::I64),
        ]))),
        Value::object(FxHashMap::from_iter([
            (name, Owned::from_value(Value::string("alice"))),
            (age, Owned::from_value(Value::int(30))),
        ])),
    );
    let c = FxHashMap::from_iter([
        (i.intern("user"), user),
        (i.intern("output"), typed(Ty::String, Value::string(""))),
    ]);
    let src = r#"
        let age = @user.age;
        @output = "unknown".to_string();
        if let true = age >= 65 { @output = "senior".to_string(); };
        if let true = age >= 18 { @output = "adult".to_string(); };
        if let true = age < 18 { @output = "minor".to_string(); };
        @output = @user.name + " (" + @output + ")";
        @output
    "#;
    let v = run_script(&i, src, c, Ty::String).await;
    assert!(v.is_string());
    assert_eq!(unsafe { v.as_str() }, "alice (adult)");
}
