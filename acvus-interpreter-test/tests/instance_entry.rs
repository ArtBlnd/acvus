//! Which declared instances a requirement can reach (RFC-0067 Decision 1).

use acvus_extern::{Externs, Interner, QualifiedRef};
use acvus_interpreter::AcvusRuntime;

struct Signature {
    namespace: &'static str,
    name: &'static str,
}

impl Signature {
    const fn new(namespace: &'static str, name: &'static str) -> Self {
        Self { namespace, name }
    }

    fn qref(&self, i: &Interner) -> QualifiedRef {
        QualifiedRef::qualified(i.intern(self.namespace), i.intern(self.name))
    }

    fn written(&self) -> String {
        format!("{}::{}", self.namespace, self.name)
    }
}

const SIGNATURES: &[Signature] = &[
    Signature::new("core", "clone"),
    Signature::new("core", "eq"),
    Signature::new("core", "hash"),
    Signature::new("core", "to_string"),
    Signature::new("core", "to_int"),
    Signature::new("iter", "into_iter"),
    Signature::new("iter", "as_iter"),
    Signature::new("num", "abs"),
    Signature::new("num", "min"),
    Signature::new("num", "max"),
    Signature::new("num", "clamp"),
    Signature::new("num", "pow"),
    Signature::new("num", "signum"),
    Signature::new("std", "vec"),
    Signature::new("vec", "filled"),
];

fn instances_without_a_mono_glue(i: &Interner) -> Vec<String> {
    let externs = Externs::<AcvusRuntime>::combine(acvus_ext::std_registries(), i)
        .expect("the standard registries combine");
    SIGNATURES
        .iter()
        .flat_map(|signature| {
            let handlers = externs
                .handlers
                .get(&signature.qref(i))
                .unwrap_or_else(|| panic!("{} is declared", signature.written()));
            handlers
                .iter()
                .enumerate()
                .filter(|(_, h)| h.is_sync() && h.instance().is_none())
                .map(move |(at, _)| format!("{}#{at}", signature.written()))
        })
        .collect()
}

#[test]
fn every_synchronous_instance_of_a_shared_signature_has_a_mono_glue() {
    let i = Interner::new();
    assert_eq!(
        instances_without_a_mono_glue(&i),
        // A receiver is one value in `ctx` and the language's `&str` is two.
        vec!["core::to_string#12".to_owned()],
        "these instances hold a site table or a state, so they are no plain \
         function and a requirement cannot be resolved to one"
    );
}
