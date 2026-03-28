mod conversion;
mod datetime;
mod deque;
mod encoding;
mod iterator;
mod list;
mod option;
mod regex;
mod sequence;
mod string;

pub use conversion::conversion_registry;
pub use datetime::datetime_registry;
pub use encoding::encoding_registry;
pub use list::list_registry;
pub use option::option_registry;
pub use regex::regex_registry;
pub use string::string_registry;

use acvus_interpreter::ExternRegistry;
use acvus_mir::ty::TypeRegistry;
use acvus_utils::Interner;

/// Register all standard library ExternFn registries.
/// Handles UserDefined type registration internally.
pub fn std_registries(interner: &Interner, type_registry: &mut TypeRegistry) -> Vec<ExternRegistry> {
    vec![
        string::string_registry(),
        conversion::conversion_registry(),
        list::list_registry(),
        option::option_registry(),
        iterator::iterator_registry(interner, type_registry),
        sequence::sequence_registry(interner, type_registry),
        deque::deque_registry(interner),
    ]
}
