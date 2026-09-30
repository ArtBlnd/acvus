//! What a call of a declared function moves between the positions of its
//! arguments and its result (ownership §5, extern-fn §4), read off the
//! declaration's lifetimes at the call's concrete types. An extern fn is
//! only called, never a value (extern-fn §1), so its lifetimes are read
//! here, at its calls, and never carried in a function type.

use crate::ty::{Concrete, Declared, FlowEnd, Ty, Types};

use super::Declarations;

/// For each output end of a call (its result, and each parameter through
/// what the call may write into), and for each position of that end, the
/// input positions it takes, each as (parameter, position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallFlows {
    pub into: Vec<(FlowEnd, Box<[Box<[(usize, usize)]>]>)>,
}

/// The flows of a call of `declared` at its concrete parameter and result
/// types: `None` where the declaration states no lifetimes, which is every
/// flow (extern-fn §4).
pub fn call_flows(decls: &Declarations, declared: Declared, types: &Types<Concrete>, params: &[Ty], ret: Ty) -> Option<CallFlows> {
    todo!("flows cut: the declaration's labels mapped onto the call's positions ({decls:p} {declared:?} {types:p} {params:?} {ret:?})")
}
