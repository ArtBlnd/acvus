//! A recorded identity argument is read back only when its index is in the
//! identity space (`0..=u32::MAX - 1`); a record outside it is refused when
//! it is read, so `SerTy::to_ty` never sees one.

use acvus_mir::ser_ty::SerTy;
use acvus_mir::ty::{IdentityId, IdentityTerm, Ty};
use acvus_utils::{Interner, LocalIdOps, QualifiedRef};

/// A user-defined type's record whose one identity argument is `index`.
fn record(index: u64) -> String {
    let i = Interner::new();
    let ty = Ty::UserDefined {
        id: QualifiedRef::root(i.intern("Held")),
        type_args: vec![],
        effect_args: vec![],
        identity_args: vec![IdentityTerm::Known(IdentityId::from_raw(0))],
        region_params: 0,
    };
    let mut json = serde_json::to_value(ty.to_ser(&i)).expect("serialize");
    json["identity_args"] = serde_json::json!([index]);
    json.to_string()
}

#[test]
fn an_identity_below_the_boundary_reads_back() {
    let i = Interner::new();
    let last = u64::from(u32::MAX - 1);
    let ser: SerTy = serde_json::from_str(&record(last)).expect("u32::MAX - 1 is an identity");
    let Ty::UserDefined { identity_args, .. } = ser.to_ty(&i) else {
        panic!("a user-defined record reads as a user-defined type");
    };
    assert_eq!(
        identity_args,
        vec![IdentityTerm::Known(IdentityId::from_raw(last as usize))]
    );
}

#[test]
fn an_identity_at_the_boundary_is_refused() {
    let json = record(u64::from(u32::MAX));
    let refusal =
        serde_json::from_str::<SerTy>(&json).expect_err("u32::MAX is outside the identity space");
    assert!(
        refusal
            .to_string()
            .contains("an identity index is at most u32::MAX - 1"),
        "{refusal}"
    );
}

#[test]
fn an_identity_above_u32_is_refused() {
    let json = record(u64::from(u32::MAX) + 1);
    assert!(serde_json::from_str::<SerTy>(&json).is_err(), "{json}");
}
