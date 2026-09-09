//! Option operations. All pure, polymorphic.

use acvus_extern::{
    ExternError, ExternRegistry, Interner, Runtime, TyVar, extern_fn, extern_registry,
};

#[extern_fn(effect = pure)]
fn unwrap<T>(_: &Interner, val: Option<T>) -> Result<T, ExternError>
where
    T: TyVar,
{
    val.ok_or_else(|| ExternError::call("unwrap", "called on None"))
}

#[extern_fn(effect = pure)]
fn unwrap_or<T>(_: &Interner, val: Option<T>, default: T) -> T
where
    T: TyVar,
{
    val.unwrap_or(default)
}

pub fn option_registry<R: Runtime>() -> ExternRegistry<R> {
    extern_registry! {
        fns: [unwrap, unwrap_or],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{TypeRegistry, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = option_registry::<TypesOnly>().register(&i, &mut TypeRegistry::new());
        assert_eq!(reg.functions.len(), 2);
        assert_eq!(reg.handlers.len(), 2);
    }
}
