//! The instances of the core signatures at the word types: `Int`, `Float`,
//! `Bool`, `Byte`, `Char` and `String` (RFC-0070 rule 5).

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

/// `cmp_float` states no `total_order`, and that is a decision. Its
/// `total_cmp` is a total order whose `Equal` is bit equality, the `==` of a
/// `Float`, so the law would hold; whether a law is stated over floats is
/// left to the owner's decision on floats.
macro_rules! instances_of {
    ($(
        $t:ty => eq: $eq:ident, clone: $clone:ident, cmp: $cmp:ident $(($($total_order:tt)*))?,
            hash: $hash:ident
    );* $(;)?) => {$(
        #[extern_fn(instance_of = acvus_extern::core::eq, effect = pure, total)]
        fn $eq(a: &$t, b: &$t) -> bool {
            Word::equals(a, b)
        }

        #[extern_fn(instance_of = acvus_extern::core::clone, effect = pure, total, copies(a))]
        fn $clone(a: &$t) -> $t {
            a.clone()
        }

        #[extern_fn(instance_of = acvus_extern::core::cmp, effect = pure, total, $($($total_order)*)?)]
        fn $cmp(a: &$t, b: &$t) -> i64 {
            verdict(Word::order(a, b))
        }

        #[extern_fn(instance_of = acvus_extern::core::hash, effect = pure, total)]
        fn $hash(a: &$t) -> u64 {
            Word::digest(a)
        }
    )*};
}

instances_of! {
    i64 => eq: eq_int, clone: clone_int, cmp: cmp_int (law(total_order)), hash: hash_int;
    f64 => eq: eq_float, clone: clone_float, cmp: cmp_float, hash: hash_float;
    bool => eq: eq_bool, clone: clone_bool, cmp: cmp_bool (law(total_order)), hash: hash_bool;
    u8 => eq: eq_byte, clone: clone_byte, cmp: cmp_byte (law(total_order)), hash: hash_byte;
    char => eq: eq_char, clone: clone_char, cmp: cmp_char (law(total_order)), hash: hash_char;
}

#[extern_fn(instance_of = acvus_extern::core::eq, effect = pure, total)]
fn eq_string(a: &String, b: &String) -> bool {
    Word::equals(a, b)
}

/// Not `total`: the clone allocates, and an allocation Rust cannot make
/// ends the process through `handle_alloc_error`.
#[extern_fn(instance_of = acvus_extern::core::clone, effect = pure, copies(a))]
fn clone_string(a: &String) -> String {
    a.clone()
}

#[extern_fn(instance_of = acvus_extern::core::cmp, effect = pure, total, law(total_order))]
fn cmp_string(a: &String, b: &String) -> i64 {
    verdict(Word::order(a, b))
}

#[extern_fn(instance_of = acvus_extern::core::hash, effect = pure, total)]
fn hash_string(a: &String) -> u64 {
    Word::digest(a)
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

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Copies, Externs, FnKind, Interner, Laws, PolyTy, QualifiedRef, TypesOnly};

    /// A fixed-seed linear congruential sequence (Knuth's MMIX constants),
    /// so a failing sample names the same inputs on every run.
    fn samples(count: usize) -> Vec<u64> {
        let mut state: u64 = 0x5eed_1a55_0c1a_7e00;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                state
            })
            .collect()
    }

    /// RFC-0082 rule 10 sampled: the sign of `cmp` is antisymmetric,
    /// transitive and total, and `0` exactly where `eq` holds.
    fn total_order_holds<T>(words: &[T], cmp: fn(&T, &T) -> i64, eq: fn(&T, &T) -> bool)
    where
        T: std::fmt::Debug,
    {
        for a in words {
            for b in words {
                let ab = cmp(a, b);
                assert!([-1, 0, 1].contains(&ab), "the sign at {a:?}, {b:?}");
                assert_eq!(ab, -cmp(b, a), "antisymmetric at {a:?}, {b:?}");
                assert_eq!(ab == 0, eq(a, b), "equal values are one value at {a:?}, {b:?}");
                for c in words.iter().take(16) {
                    if ab <= 0 && cmp(b, c) <= 0 {
                        assert!(cmp(a, c) <= 0, "transitive at {a:?}, {b:?}, {c:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn total_order_holds_over_the_declared_cmp_instances() {
        let raw = samples(48);
        let ints: Vec<i64> = [i64::MIN, i64::MAX, 0, 1, -1]
            .into_iter()
            .chain(raw.iter().map(|&word| word as i64))
            .collect();
        let bytes: Vec<u8> = (0..=u8::MAX).collect();
        let chars: Vec<char> = ['\0', 'a', 'z', 'é', '\u{10FFFF}']
            .into_iter()
            .chain(raw.iter().filter_map(|&word| char::from_u32((word >> 43) as u32)))
            .collect();
        let strings: Vec<String> = ["", "a", "ab", "b", "é", "a\0"]
            .into_iter()
            .map(str::to_string)
            .chain(raw.iter().map(|word| format!("{:x}", word % 4096)))
            .collect();
        total_order_holds(&ints, cmp_int, eq_int);
        total_order_holds(&[false, true], cmp_bool, eq_bool);
        total_order_holds(&bytes, cmp_byte, eq_byte);
        total_order_holds(&chars, cmp_char, eq_char);
        total_order_holds(&strings, cmp_string, eq_string);
    }

    /// RFC-0082 rule 10 sampled: each clone is the value its argument lends,
    /// under the type's own `==` (bit equality on a `Float`).
    #[test]
    fn copies_holds_over_the_declared_clone_instances() {
        for word in samples(64) {
            let int = word as i64;
            assert!(eq_int(&clone_int(&int), &int));
            let float = f64::from_bits(word);
            assert!(eq_float(&clone_float(&float), &float), "{float:?}");
            let byte = word as u8;
            assert!(eq_byte(&clone_byte(&byte), &byte));
            let text = format!("{word:x}é");
            assert!(eq_string(&clone_string(&text), &text));
            if let Some(c) = char::from_u32((word >> 43) as u32) {
                assert!(eq_char(&clone_char(&c), &c));
            }
        }
        for b in [false, true] {
            assert!(eq_bool(&clone_bool(&b), &b));
        }
        for special in [f64::NAN, -0.0, 0.0, f64::INFINITY] {
            assert!(eq_float(&clone_float(&special), &special), "{special:?}");
        }
    }

    /// The declarations are the instances': every clone copies its one
    /// argument, and every `cmp` states `total_order` but the `Float`'s.
    #[test]
    fn copies_and_total_order_are_declared_by_instance() {
        let i = Interner::new();
        let reg = Externs::combine(vec![word_registry::<TypesOnly>()], &i)
            .expect("registries combine");
        let instances_of = |name: &str| {
            let qref = QualifiedRef::qualified(i.intern("core"), i.intern(name));
            let function = reg
                .functions
                .iter()
                .find(|f| f.qref == qref)
                .expect("the signature is declared");
            let FnKind::Extern { instances, .. } = &function.kind else {
                panic!("{name} is an extern")
            };
            instances.concrete.clone()
        };
        let clones = instances_of("clone");
        assert_eq!(clones.len(), 6);
        for instance in &clones {
            assert_eq!(instance.copies, Some(Copies { param: 0 }), "{:?}", instance.ty);
        }
        let cmps = instances_of("cmp");
        assert_eq!(cmps.len(), 6);
        for instance in &cmps {
            let PolyTy::Fn { params, .. } = &instance.ty else {
                panic!("an instance is a function")
            };
            let over_float = matches!(&params[0].ty, PolyTy::Ref(_, lent) if *lent.ty() == PolyTy::Float);
            let expected = match over_float {
                true => Laws::None,
                false => Laws::TotalOrder,
            };
            assert_eq!(instance.laws, expected, "{:?}", instance.ty);
        }
    }
}
