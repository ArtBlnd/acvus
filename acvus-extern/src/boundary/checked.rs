//! The checking adapter's signature (K15): `Checked` controls every guard it
//! hands out, each a lock on a slot its storage keeps at a stable place, and by
//! K12 the extern side relies on storage only while a guard lends it.

use crate::boundary::signed::{SignedS1, SignedS2, SignedS3};
use crate::checked::{Checked, CheckedStorage};

// SAFETY: to be argued by the fill, against the adapter's guards and storage.
unsafe impl<C: CheckedStorage> SignedS1 for Checked<C> {}
unsafe impl<C: CheckedStorage> SignedS2 for Checked<C> {}
unsafe impl<C: CheckedStorage> SignedS3 for Checked<C> {}
