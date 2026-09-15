//! A context read through references while temporaries are built from it (RFC-0018).

use acvus_interpreter::Value;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[tokio::test]
async fn a_context_string_rebuilt_from_itself_through_temporaries() {
    let i = Interner::new();
    let name = i.intern("name");
    let age = i.intern("age");
    let user = typed(
        Ty::Object(FxHashMap::from_iter([(name, Ty::String), (age, Ty::Int)])),
        Value::object(FxHashMap::from_iter([(name, Value::string("alice")), (age, Value::int(30))])),
    );
    let c = FxHashMap::from_iter([
        (i.intern("user"), user),
        (i.intern("output"), typed(Ty::String, Value::string(""))),
    ]);
    let src = r#"
        age = @user.age;
        @output = "unknown";
        true = age >= 65 { @output = "senior"; };
        true = age >= 18 { @output = "adult"; };
        true = age < 18 { @output = "minor"; };
        @output = @user.name + " (" + @output + ")";
        @output
    "#;
    let v = run_script(&i, src, c).await;
    assert!(v.is_string());
    assert_eq!(unsafe { v.as_str() }, "alice (adult)");
}
