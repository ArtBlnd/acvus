//! Generic functions, impls and requirements (impls §1-§3).

use crate::boundary::entry::Settled;
use crate::boundary::instance::InstanceEntry;
use crate::runtime::Runtime;

/// The task an impl entry is read back at.
pub mod task {
    mod sealed {
        pub trait Sealed {}
    }

    pub trait TaskKind: sealed::Sealed + 'static {}

    /// Called and returning.
    pub struct Returning;
    /// Called and awaited.
    pub struct Awaited;

    impl sealed::Sealed for Returning {}
    impl sealed::Sealed for Awaited {}
    impl TaskKind for Returning {}
    impl TaskKind for Awaited {}
}

use task::TaskKind;

/// How a generic function takes its receiver, which picks the requirement's
/// form (impls §3).
pub mod receiver {
    mod sealed {
        pub trait Sealed {}
    }

    pub trait ReceiverMode: sealed::Sealed {}

    /// By value or `&mut`: the requirer owns the receiver with its impl.
    pub struct Steps;
    /// By `&`: the requirer holds the impl at the type.
    pub struct Reads;

    impl sealed::Sealed for Steps {}
    impl sealed::Sealed for Reads {}
    impl ReceiverMode for Steps {}
    impl ReceiverMode for Reads {}
}

/// The nominal marker of one generic function, with no marker arguments: its
/// receiver mode and its function type at each task.
pub trait Family: 'static {
    type Receiver: receiver::ReceiverMode;
    /// The receiver's type at the requirer's type arguments `Ts`.
    type ReceiverAt<Ts>;
    /// The parameters after the receiver, lent to a returning call.
    type Rest<'l, R: Runtime>;
    /// The same parameters, moved into an awaited call (K3).
    type RestMoved<R: Runtime>;
    /// The result; it may borrow the receiver `'r` where the function's
    /// flows say so.
    type Output<'r, R: Runtime>;
    /// The glue's function pointer, typed at the function type for task `K`:
    /// it takes the receiver as the family lends it and the rest, and gives
    /// the result as a runtime value, which the requirer reads at its own type
    /// through the runtime (impls §3).
    type Glue<R: Runtime, K: TaskKind>: Copy + Send + Sync + 'static;
}

/// Making a value from an impl entry and reading an entry back (runtime
/// contract §2).
pub trait Instances: Runtime {
    fn instance<F, K>(&self, entry: InstanceEntry<Self, F, K>) -> Self::Value
    where
        F: Family,
        K: TaskKind;
    fn entry<F, K>(&self, v: Self::Value, at: Settled<'_>) -> &InstanceEntry<Self, F, K>
    where
        F: Family,
        K: TaskKind;
    fn task_of(&self, v: Self::Value, at: Settled<'_>) -> Task;
}

/// An impl value's task, read before an awaited call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Returning,
    Awaited,
}
