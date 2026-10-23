use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

/// The effect is `opaque` rather than the cheaper two because the line
/// leaves the machine. `pure` would let the optimizer drop a print whose
/// result nothing reads, and `idempotent` would let it keep one of two
/// identical prints; the order chain RFC-0007 builds over opaque calls is
/// what puts the lines out in the order the script wrote them.
#[extern_fn(effect = opaque)]
fn print(s: &str) {
    println!("{s}");
}

pub fn io_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "io",
        fns: [print],
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
            Externs::combine(vec![io_registry::<TypesOnly>()], &i).expect("registry combines");
        let core = Externs::<TypesOnly>::combine(vec![], &i).expect("core combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 1);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 1);
    }
}
