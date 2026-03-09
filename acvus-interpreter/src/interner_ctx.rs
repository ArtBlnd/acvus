//! Thread-local interner context for the interpreter crate.
//!
//! Used by `IntoValue<Option>`, `FromValue<Option>`, and `builtin_unwrap`
//! which need an interner to intern "Some"/"None" variant tags.
//! The interpreter sets this before execution via `set_interner`.

use acvus_utils::Interner;
use std::cell::RefCell;

thread_local! {
    static INTERNER: RefCell<Option<Interner>> = const { RefCell::new(None) };
}

pub(crate) fn set_interner(interner: &Interner) {
    INTERNER.with(|cell| {
        *cell.borrow_mut() = Some(interner.clone());
    });
}

pub(crate) fn get_interner() -> Option<Interner> {
    INTERNER.with(|cell| cell.borrow().clone())
}
