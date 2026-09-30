//! The adapter's signatures (K15). The adapter lends X's storage through X's
//! guards and makes and reads X's values through X's operations; what it adds
//! only observes them and panics. So each fact X signs holds of the adapter.

use crate::adapter::Adapter;
use crate::runtime::Runtime;
use crate::boundary::signed::{SignedS1, SignedS2, SignedS3};

// SAFETY: every guard the adapter hands out is X's, and the adapter keeps no
// storage of its own; X's S1 is the adapter's.
unsafe impl<X: Runtime + SignedS1> SignedS1 for Adapter<X> {}
// SAFETY: every reference value the adapter makes is X's, made by X's own
// operation; X's S2 is the adapter's.
unsafe impl<X: Runtime + SignedS2> SignedS2 for Adapter<X> {}
// SAFETY: the adapter releases nothing itself; X's S3 is the adapter's.
unsafe impl<X: Runtime + SignedS3> SignedS3 for Adapter<X> {}
