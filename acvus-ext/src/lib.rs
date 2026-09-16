mod container;
mod conversion;
mod datetime;
mod decimal;
mod deque;
mod encoding;
mod iterator;
mod option;
mod panic;
mod regex;
mod string;
mod vec;

pub use container::container_registry;
pub use conversion::{conversion_registry, from_str_registries};
pub use datetime::datetime_registry;
pub use decimal::{Decimal, decimal_registry};
pub use deque::{Deque, deque_registry};
pub use encoding::encoding_registry;
pub use iterator::iterator_registry;
pub use option::option_registry;
pub use panic::panic_registry;
pub use regex::regex_registry;
pub use string::string_registry;
pub use vec::vec_registry;

use acvus_extern::{Registry, Runtime};

/// The standard registries. Each registers its own types when registered.
pub fn std_registries<R>() -> Vec<Registry<R>>
where
    R: Runtime,
{
    let mut registries = vec![
        string_registry(),
        conversion_registry(),
        decimal_registry(),
        vec_registry(),
        container_registry(),
        deque_registry(),
        option_registry(),
        panic_registry(),
        iterator_registry(),
    ];
    registries.extend(from_str_registries());
    registries
}
