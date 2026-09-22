//! The instances of the core signatures at the word types: `Int`, `Float`,
//! `Bool`, `Byte`, `Char` and `String` (RFC-0070 D5).

use std::cmp::Ordering;
use std::hash::{DefaultHasher, Hash, Hasher};

use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

/// The answer `core::cmp` gives, which `acvus-mir` lowers `<`, `<=`, `>`
/// and `>=` to the sign of.
pub(crate) fn verdict(order: Ordering) -> i64 {
    match order {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

trait Word: Clone + Send + Sync + 'static {
    fn equals(&self, other: &Self) -> bool;
    fn order(&self, other: &Self) -> Ordering;
    fn digest(&self) -> u64;
}

macro_rules! word_by_ord {
    ($($t:ty => |$value:ident| $digest:block)*) => {$(
        impl Word for $t {
            fn equals(&self, other: &Self) -> bool {
                self == other
            }

            fn order(&self, other: &Self) -> Ordering {
                Ord::cmp(self, other)
            }

            fn digest(&self) -> u64 {
                let $value = self;
                $digest
            }
        }
    )*};
}

word_by_ord! {
    i64 => |n| { *n as u64 }
    u8 => |b| { u64::from(*b) }
    bool => |b| { u64::from(*b) }
    char => |c| { u64::from(*c) }
    String => |s| {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

/// An obligation across artifacts: the interpreter's `==` on a `Float` is
/// bit equality and its `<` is `total_cmp` (RFC-0020), and a requirement
/// site reaching these instances must see the answers those instructions
/// give. Bit equality is also what makes a `Float` hashable at all: `0.0`
/// and `-0.0` are two keys here, as they are two bit patterns.
impl Word for f64 {
    fn equals(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }

    fn order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }

    fn digest(&self) -> u64 {
        self.to_bits()
    }
}

macro_rules! instances_of {
    ($(
        $t:ty => eq: $eq:ident, clone: $clone:ident, cmp: $cmp:ident, hash: $hash:ident
    );* $(;)?) => {$(
        #[extern_fn(instance_of = acvus_extern::core::eq, effect = pure)]
        fn $eq(a: &$t, b: &$t) -> bool {
            Word::equals(a, b)
        }

        #[extern_fn(instance_of = acvus_extern::core::clone, effect = pure)]
        fn $clone(a: &$t) -> $t {
            a.clone()
        }

        #[extern_fn(instance_of = acvus_extern::core::cmp, effect = pure)]
        fn $cmp(a: &$t, b: &$t) -> i64 {
            verdict(Word::order(a, b))
        }

        #[extern_fn(instance_of = acvus_extern::core::hash, effect = pure)]
        fn $hash(a: &$t) -> u64 {
            Word::digest(a)
        }
    )*};
}

instances_of! {
    i64 => eq: eq_int, clone: clone_int, cmp: cmp_int, hash: hash_int;
    f64 => eq: eq_float, clone: clone_float, cmp: cmp_float, hash: hash_float;
    bool => eq: eq_bool, clone: clone_bool, cmp: cmp_bool, hash: hash_bool;
    u8 => eq: eq_byte, clone: clone_byte, cmp: cmp_byte, hash: hash_byte;
    char => eq: eq_char, clone: clone_char, cmp: cmp_char, hash: hash_char;
    String => eq: eq_string, clone: clone_string, cmp: cmp_string, hash: hash_string;
}

pub fn word_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "core",
        fns: [
            eq_int, clone_int, cmp_int, hash_int,
            eq_float, clone_float, cmp_float, hash_float,
            eq_bool, clone_bool, cmp_bool, hash_bool,
            eq_byte, clone_byte, cmp_byte, hash_byte,
            eq_char, clone_char, cmp_char, hash_char,
            eq_string, clone_string, cmp_string, hash_string,
        ],
    }
}
