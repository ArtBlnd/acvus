//! Option operations. All pure, polymorphic.

use acvus_extern::{ExternError, Registry, Runtime, TyVar, extern_fn, extern_registry};

#[extern_fn(effect = pure)]
fn unwrap<T>(val: Option<T>) -> Result<T, ExternError>
where
    T: TyVar,
{
    val.ok_or_else(|| ExternError::call("unwrap", "called on None"))
}

#[extern_fn(effect = pure)]
fn unwrap_or<T>(val: Option<T>, default: T) -> T
where
    T: TyVar,
{
    val.unwrap_or(default)
}

pub fn option_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        fns: [unwrap, unwrap_or],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg =
            Externs::combine(vec![option_registry::<TypesOnly>()], &i).expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 2);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 2);
    }
}
