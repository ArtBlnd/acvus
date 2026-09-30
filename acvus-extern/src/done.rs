//! A call is done only by the body its entry opened (crossing §1, §4).

use crate::runtime::Runtime;

/// The result of one call: the runtime value its context made at the site's
/// settled result type, or a reference made from the guard a result borrows.
pub struct Done<R: Runtime> {
    value: R::Value,
}

impl<R: Runtime> Done<R> {
    pub(crate) fn of(value: R::Value) -> Self {
        Done { value }
    }

    /// The runtime's view of the result; only a runtime reads it.
    pub fn into_value(self) -> R::Value {
        self.value
    }
}
