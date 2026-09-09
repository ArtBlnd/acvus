//! Option operations. All pure, polymorphic.

use acvus_extern::{ExternRegistry, Interner, RuntimeError, TyVar, extern_fn, extern_registry};

#[extern_fn(effect = pure)]
fn unwrap<T: TyVar>(_: &Interner, val: Option<T>) -> Result<T, RuntimeError> {
    val.ok_or_else(|| RuntimeError::extern_call("unwrap", "called on None"))
}

#[extern_fn(effect = pure)]
fn unwrap_or<T: TyVar>(_: &Interner, val: Option<T>, default: T) -> T {
    val.unwrap_or(default)
}

pub fn option_registry() -> ExternRegistry {
    extern_registry! {
        fns: [unwrap, unwrap_or],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::TypeRegistry;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = option_registry().register(&i, &mut TypeRegistry::new());
        assert_eq!(reg.functions.len(), 2);
        assert_eq!(reg.executables.len(), 2);
    }
}
