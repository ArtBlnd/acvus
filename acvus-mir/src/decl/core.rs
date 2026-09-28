//! `core`, declared inside the compiler (operators §2): the names of its
//! generic functions, the lang items the language's syntax lowers to, and
//! the tags of the primitive `Option` and `Result` (types §2). One place
//! for every reader: typeck, laws, fusion.

use acvus_utils::{Astr, Interner, QualifiedRef};

/// The names of `core`'s generic functions: the lang items (operators §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Core {
    pub add: QualifiedRef,
    pub sub: QualifiedRef,
    pub mul: QualifiedRef,
    pub div: QualifiedRef,
    pub rem: QualifiedRef,
    pub neg: QualifiedRef,
    pub eq: QualifiedRef,
    pub cmp: QualifiedRef,
    pub as_slice: QualifiedRef,
    pub as_slice_mut: QualifiedRef,
    pub as_str: QualifiedRef,
    pub display: QualifiedRef,
    pub clone: QualifiedRef,
    pub unwrap: QualifiedRef,
    pub tags: Tags,
}

/// The variants of the primitive `Option` and `Result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tags {
    pub some: Astr,
    pub none: Astr,
    pub ok: Astr,
    pub err: Astr,
}

impl Core {
    pub fn of(interner: &Interner) -> Core {
        let namespace = interner.intern("core");
        let name = |name: &str| QualifiedRef::qualified(namespace, interner.intern(name));
        Core {
            add: name("add"),
            sub: name("sub"),
            mul: name("mul"),
            div: name("div"),
            rem: name("rem"),
            neg: name("neg"),
            eq: name("eq"),
            cmp: name("cmp"),
            as_slice: name("as_slice"),
            as_slice_mut: name("as_slice_mut"),
            as_str: name("as_str"),
            display: name("display"),
            clone: name("clone"),
            unwrap: name("unwrap"),
            tags: Tags {
                some: interner.intern("Some"),
                none: interner.intern("None"),
                ok: interner.intern("Ok"),
                err: interner.intern("Err"),
            },
        }
    }
}
