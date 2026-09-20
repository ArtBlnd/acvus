mod array;
mod char;
mod conversion;
mod datetime;
mod decimal;
mod deque;
mod encoding;
mod hash;
mod io;
mod iter;
mod iterator;
mod num;
mod option;
mod panic;
mod regex;
mod result;
mod string;
mod vec;

pub use array::array_registry;
pub use char::char_registry;
pub use conversion::{conversion_registry, from_str_registries};
pub use datetime::datetime_registry;
pub use decimal::{Decimal, decimal_registry};
pub use deque::{Deque, deque_registry};
pub use encoding::encoding_registry;
pub use hash::hash_registry;
pub use io::io_registry;
pub use iter::Iter;
pub use iterator::iterator_registry;
pub use num::num_registry;
pub use option::option_registry;
pub use panic::panic_registry;
pub use regex::regex_registry;
pub use result::result_registry;
pub use string::string_registry;
pub use vec::vec_registry;

use acvus_extern::{Registry, Runtime};

/// The standard registries. Each registers its own types when registered.
///
/// `regex_registry`, `datetime_registry`, `encoding_registry` and
/// `io_registry` are deliberately not here. They are domain libraries, each
/// with a crate or a host resource behind it, and a host adds the ones it
/// wants; this list is the surface every host gets whether it asked or not,
/// so growing it moves the overload set of every program already written
/// against it.
pub fn std_registries<R>() -> Vec<Registry<R>>
where
    R: Runtime,
{
    let mut registries = vec![
        string_registry(),
        char_registry(),
        conversion_registry(),
        decimal_registry(),
        vec_registry(),
        array_registry(),
        deque_registry(),
        option_registry(),
        result_registry(),
        panic_registry(),
        iterator_registry(),
        num_registry(),
        hash_registry(),
    ];
    registries.extend(from_str_registries());
    registries
}
