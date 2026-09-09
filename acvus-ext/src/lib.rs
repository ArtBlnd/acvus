mod conversion;
mod datetime;
mod encoding;
mod iter_pipeline;
mod iterator;
mod list;
mod option;
mod regex;
mod string;

pub use conversion::conversion_registry;
pub use datetime::datetime_registry;
pub use encoding::encoding_registry;
pub use iter_pipeline::{Iter, IterHandle};
pub use iterator::iterator_registry;
pub use list::{List, list_registry, list_ty, list_value, sequence_items};
pub use option::option_registry;
pub use regex::regex_registry;
pub use string::string_registry;

use acvus_extern::ExternRegistry;

/// The standard registries. Each registers its own types when registered.
pub fn std_registries() -> Vec<ExternRegistry> {
    vec![
        string_registry(),
        conversion_registry(),
        list_registry(),
        option_registry(),
        iterator_registry(),
    ]
}
