mod conversion;
mod datetime;
mod deque;
mod encoding;
mod iter_pipeline;
mod iterator;
mod list;
mod option;
mod regex;
mod string;

pub use conversion::conversion_registry;
pub use datetime::datetime_registry;
pub use deque::deque_registry;
pub use encoding::encoding_registry;
pub use iter_pipeline::Iter;
pub use iterator::iterator_registry;
pub use list::{List, list_registry, list_ty};
pub use option::option_registry;
pub use regex::regex_registry;
pub use string::string_registry;

use acvus_extern::{ExternRegistry, Runtime};

/// The standard registries. Each registers its own types when registered.
pub fn std_registries<R: Runtime>() -> Vec<ExternRegistry<R>> {
    vec![
        string_registry(),
        conversion_registry(),
        list_registry(),
        deque_registry(),
        option_registry(),
        iterator_registry(),
    ]
}
